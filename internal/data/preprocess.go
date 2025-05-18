package data

import (
	"regexp"
	"strings"
)

func CleanText(raw string) string {
	clean := strings.ToLower(raw)
	clean = regexp.MustCompile(`https?://\S+`).ReplaceAllString(clean, "")
	clean = regexp.MustCompile(`[^\w\s]`).ReplaceAllString(clean, "")
	clean = strings.TrimSpace(clean)
	return clean
}
