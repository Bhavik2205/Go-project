package auth

import (
	"net/http"

	"github.com/Bhavik2205/ML-Bot/internal/auth"
	"github.com/Bhavik2205/ML-Bot/internal/validation"
)

type refreshRequest struct {
	RefreshToken string `json:"refreshToken" validate:"required"`
}

type refreshResponse struct {
	AccessToken  string `json:"accessToken"`
	RefreshToken string `json:"refreshToken"`
	ExpiresIn    int    `json:"expiresIn"`
}

func HandleRefresh() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req refreshRequest
		if !validation.BindAndValidate(w, r, &req) {
			return
		}

		claims, err := auth.ParseToken(req.RefreshToken, auth.TokenTypeRefresh)
		if err != nil {
			writeError(w, http.StatusUnauthorized, r, "UNAUTHORIZED", "Invalid or expired refresh token", nil)
			return
		}

		accessToken, refreshToken, err := generateTokenPair(claims.UserID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, r, "INTERNAL_ERROR", "Failed to generate tokens", nil)
			return
		}

		writeSuccess(w, http.StatusOK, r, refreshResponse{
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    900,
		})
	}
}
