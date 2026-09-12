// Package models reports the state of the ML model assets and the sentiment
// pipeline that consumes them.
//
// It deliberately reports only facts the system can actually observe — which
// model files exist, whether inference is compiled in, how much of the news
// corpus has been scored. It does not report accuracy or F1, because nothing
// in this system measures them; publishing invented numbers would be worse
// than publishing none.
package models

import (
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/Bhavik2205/ML-Bot/internal/api/handlers/httpx"
	"github.com/Bhavik2205/ML-Bot/internal/db"
)

// modelDir is where ONNX assets live, relative to the working directory.
const modelDir = "models"

type modelFileDTO struct {
	Name       string    `json:"name"`
	SizeBytes  int64     `json:"size_bytes"`
	ModifiedAt time.Time `json:"modified_at"`
	Present    bool      `json:"present"`
}

type pipelineDTO struct {
	TotalArticles  int64      `json:"total_articles"`
	ScoredArticles int64      `json:"scored_articles"`
	CoveragePct    float64    `json:"coverage_pct"`
	LastAnalyzedAt *time.Time `json:"last_analyzed_at"`
	LastArticleAt  *time.Time `json:"last_article_at"`
}

type responseDTO struct {
	// InferenceAvailable is false on builds without the ONNX runtime, which
	// is the normal state on Windows where the package needs cgo.
	InferenceAvailable bool           `json:"inference_available"`
	InferenceNote      string         `json:"inference_note"`
	ModelDir           string         `json:"model_dir"`
	Files              []modelFileDTO `json:"files"`
	Pipeline           pipelineDTO    `json:"pipeline"`
}

// HandleGet reports model assets and sentiment pipeline coverage.
func HandleGet(dbc *db.DBClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := responseDTO{
			ModelDir:           modelDir,
			InferenceAvailable: inferenceAvailable,
			InferenceNote:      inferenceNote,
			Files:              scanModelDir(),
		}

		if dbc != nil {
			resp.Pipeline = pipelineStats(dbc)
		}

		httpx.WriteSuccess(w, http.StatusOK, r, resp)
	}
}

// scanModelDir lists the ONNX assets on disk.
func scanModelDir() []modelFileDTO {
	out := []modelFileDTO{}

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		// A missing directory is a legitimate state to report, not an error.
		return out
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".onnx" {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out = append(out, modelFileDTO{
			Name:       e.Name(),
			SizeBytes:  info.Size(),
			ModifiedAt: info.ModTime(),
			Present:    true,
		})
	}
	return out
}

// pipelineStats measures how much of the news corpus has been scored.
func pipelineStats(dbc *db.DBClient) pipelineDTO {
	var p pipelineDTO

	dbc.DB.Model(&db.NewsArticle{}).Count(&p.TotalArticles)
	dbc.DB.Model(&db.NewsArticle{}).Where("sentiment_label <> ''").Count(&p.ScoredArticles)

	if p.TotalArticles > 0 {
		p.CoveragePct = float64(p.ScoredArticles) / float64(p.TotalArticles) * 100
	}

	var last struct {
		AnalyzedAt  *time.Time
		PublishedAt *time.Time
	}
	dbc.DB.Model(&db.NewsArticle{}).
		Select("MAX(analyzed_at) AS analyzed_at, MAX(published_at) AS published_at").
		Scan(&last)
	p.LastAnalyzedAt = last.AnalyzedAt
	p.LastArticleAt = last.PublishedAt

	return p
}
