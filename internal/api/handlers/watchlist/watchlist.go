// Package watchlist manages a user's tracked instruments.
package watchlist

import (
	"encoding/json"
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
	"github.com/Bhavik2205/ML-Bot/internal/middleware"
	"gorm.io/gorm"
)

const defaultName = "Default"

type itemDTO struct {
	ID              uint   `json:"id"`
	InstrumentToken uint32 `json:"instrument_token"`
	Symbol          string `json:"symbol"`
}

type watchlistDTO struct {
	ID    uint      `json:"id"`
	Name  string    `json:"name"`
	Items []itemDTO `json:"items"`
}

type addRequest struct {
	InstrumentToken uint32 `json:"instrument_token"`
	Symbol          string `json:"symbol"`
}

// userIDFrom reads the authenticated user from the request context.
func userIDFrom(r *http.Request) (uint, bool) {
	id := middleware.UserIDFromContext(r.Context())
	if id == 0 {
		return 0, false
	}
	return id, true
}

// ensureDefault fetches the user's default watchlist, creating it on first use
// so the client never has to handle a "no watchlist yet" case.
func ensureDefault(tx *gorm.DB, userID uint) (*db.Watchlist, error) {
	var wl db.Watchlist
	err := tx.Where("user_id = ? AND name = ?", userID, defaultName).First(&wl).Error
	if err == nil {
		return &wl, nil
	}
	if err != gorm.ErrRecordNotFound {
		return nil, err
	}

	wl = db.Watchlist{UserID: userID, Name: defaultName}
	if err := tx.Create(&wl).Error; err != nil {
		return nil, err
	}
	return &wl, nil
}

// HandleGetDefault returns the user's default watchlist with its items.
func HandleGetDefault(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		wl, err := ensureDefault(dbc.DB, userID)
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load watchlist", err.Error())
			return
		}

		var items []db.WatchlistItem
		if err := dbc.DB.Where("watchlist_id = ?", wl.ID).Order("symbol ASC").Find(&items).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load items", err.Error())
			return
		}

		out := watchlistDTO{ID: wl.ID, Name: wl.Name, Items: make([]itemDTO, 0, len(items))}
		for _, it := range items {
			out.Items = append(out.Items, itemDTO{ID: it.ID, InstrumentToken: it.InstrumentToken, Symbol: it.Symbol})
		}
		httpx.WriteSuccess(w, http.StatusOK, r, out)
	}
}

// HandleAddItem adds an instrument to the default watchlist.
func HandleAddItem(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		var req addRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest, "request body must be valid JSON", err.Error())
			return
		}
		if req.InstrumentToken == 0 || req.Symbol == "" {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"instrument_token and symbol are required", nil)
			return
		}

		wl, err := ensureDefault(dbc.DB, userID)
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load watchlist", err.Error())
			return
		}

		// Adding the same instrument twice is a no-op rather than an error:
		// the caller's intent is satisfied either way.
		var existing db.WatchlistItem
		err = dbc.DB.Where("watchlist_id = ? AND instrument_token = ?", wl.ID, req.InstrumentToken).First(&existing).Error
		if err == nil {
			httpx.WriteSuccess(w, http.StatusOK, r, itemDTO{
				ID: existing.ID, InstrumentToken: existing.InstrumentToken, Symbol: existing.Symbol,
			})
			return
		}
		if err != gorm.ErrRecordNotFound {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to check item", err.Error())
			return
		}

		item := db.WatchlistItem{WatchlistID: wl.ID, InstrumentToken: req.InstrumentToken, Symbol: req.Symbol}
		if err := dbc.DB.Create(&item).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to add item", err.Error())
			return
		}
		httpx.WriteSuccess(w, http.StatusCreated, r, itemDTO{
			ID: item.ID, InstrumentToken: item.InstrumentToken, Symbol: item.Symbol,
		})
	}
}

// HandleDeleteItem removes an instrument from the default watchlist.
func HandleDeleteItem(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}
		userID, ok := userIDFrom(r)
		if !ok {
			httpx.WriteError(w, http.StatusUnauthorized, r, httpx.CodeForbidden, "not authenticated", nil)
			return
		}

		token, ok := httpx.QueryUint32(r, "instrument_token")
		if !ok {
			httpx.WriteError(w, http.StatusBadRequest, r, httpx.CodeBadRequest,
				"instrument_token query parameter is required", nil)
			return
		}

		wl, err := ensureDefault(dbc.DB, userID)
		if err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to load watchlist", err.Error())
			return
		}

		// Scoping the delete by watchlist_id keeps one user from deleting
		// another user's row by guessing an instrument token.
		res := dbc.DB.Where("watchlist_id = ? AND instrument_token = ?", wl.ID, token).Delete(&db.WatchlistItem{})
		if res.Error != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to delete item", res.Error.Error())
			return
		}
		if res.RowsAffected == 0 {
			httpx.WriteError(w, http.StatusNotFound, r, httpx.CodeNotFound, "instrument is not in the watchlist", nil)
			return
		}
		httpx.WriteSuccess(w, http.StatusOK, r, map[string]any{"deleted": true, "instrument_token": token})
	}
}
