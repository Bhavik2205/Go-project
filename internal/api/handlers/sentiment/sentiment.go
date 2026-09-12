// Package sentiment serves news articles and their sentiment scores.
package sentiment

import (
	"net/http"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
)

type articleDTO struct {
	ID             uint      `json:"id"`
	Source         string    `json:"source"`
	Title          string    `json:"title"`
	Description    string    `json:"description"`
	URL            string    `json:"url"`
	ImageURL       string    `json:"image_url"`
	PublishedAt    time.Time `json:"published_at"`
	SentimentScore float64   `json:"sentiment_score"`
	SentimentLabel string    `json:"sentiment_label"`
	Analyzed       bool      `json:"analyzed"`
}

type summaryDTO struct {
	Total     int64   `json:"total"`
	Analyzed  int64   `json:"analyzed"`
	Positive  int64   `json:"positive"`
	Neutral   int64   `json:"neutral"`
	Negative  int64   `json:"negative"`
	AvgScore  float64 `json:"avg_score"`
	WindowHrs int     `json:"window_hours"`
}

type responseDTO struct {
	Summary  summaryDTO   `json:"summary"`
	Articles []articleDTO `json:"articles"`
}

// HandleGet returns recent articles plus an aggregate sentiment summary.
//
// Query: ?limit=<1..200>&hours=<1..720>&label=Positive|Neutral|Negative
func HandleGet(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if dbc == nil {
			httpx.WriteError(w, http.StatusServiceUnavailable, r, httpx.CodeUnavailable, "database unavailable", nil)
			return
		}

		limit := httpx.QueryInt(r, "limit", 50, 1, 200)
		hours := httpx.QueryInt(r, "hours", 24, 1, 720)
		since := time.Now().Add(-time.Duration(hours) * time.Hour)

		// Summary over the whole window, not just the returned page — a count
		// of the page would be a misleading denominator.
		var summary summaryDTO
		summary.WindowHrs = hours

		base := dbc.DB.Model(&db.NewsArticle{}).Where("published_at >= ?", since)
		base.Count(&summary.Total)

		// analyzed_at is zero until the sentiment model has scored the row.
		analyzed := dbc.DB.Model(&db.NewsArticle{}).
			Where("published_at >= ? AND sentiment_label <> ''", since)
		analyzed.Count(&summary.Analyzed)

		for _, lbl := range []struct {
			name string
			dest *int64
		}{
			{"Positive", &summary.Positive},
			{"Neutral", &summary.Neutral},
			{"Negative", &summary.Negative},
		} {
			dbc.DB.Model(&db.NewsArticle{}).
				Where("published_at >= ? AND sentiment_label = ?", since, lbl.name).
				Count(lbl.dest)
		}

		// COALESCE keeps the scan working when the window holds no scored rows.
		var avg struct{ Avg float64 }
		dbc.DB.Model(&db.NewsArticle{}).
			Select("COALESCE(AVG(sentiment_score), 0) AS avg").
			Where("published_at >= ? AND sentiment_label <> ''", since).
			Scan(&avg)
		summary.AvgScore = avg.Avg

		q := dbc.DB.Model(&db.NewsArticle{}).
			Where("published_at >= ?", since).
			Order("published_at DESC").
			Limit(limit)
		if lbl := r.URL.Query().Get("label"); lbl != "" {
			q = q.Where("sentiment_label = ?", lbl)
		}

		var rows []db.NewsArticle
		if err := q.Find(&rows).Error; err != nil {
			httpx.WriteError(w, http.StatusInternalServerError, r, httpx.CodeInternal, "failed to query news", err.Error())
			return
		}

		out := make([]articleDTO, 0, len(rows))
		for _, a := range rows {
			out = append(out, articleDTO{
				ID: a.ID, Source: a.Source, Title: a.Title, Description: a.Description,
				URL: a.URL, ImageURL: a.ImageURL, PublishedAt: a.PublishedAt,
				SentimentScore: a.SentimentScore, SentimentLabel: a.SentimentLabel,
				Analyzed: a.SentimentLabel != "",
			})
		}

		httpx.WriteSuccess(w, http.StatusOK, r, responseDTO{Summary: summary, Articles: out})
	}
}
