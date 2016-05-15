package butteraugli

import (
	"image"
	"math"
)

var kSrgbToLinearTable = func() []float64 {
	var table [256]float64
	for i, _ := range table {
		var srgb float64 = float64(i) / 255.0
		var m float64
		if srgb <= 0.04045 {
			m = srgb / 12.92
		} else {
			m = math.Pow((srgb+0.055)/1.055, 2.4)
		}
		table[i] = 255.0 * m
	}
	return table[:]
}()

// Translate R, G, B channels from sRGB to linear space. If an alpha channel
// is present, overlay the image over a black or white background. Overlaying
// is done in the sRGB space; while technically incorrect, this is aligned with
// many other software (web browsers, WebP near lossless).
func ImageToLinearOnBlack(img image.Image) ([][]float64, bool) {

	const bg = 0

	var (
		rgb      [3][]float64
		hasAlpha bool

		bnds = img.Bounds()
	)

	size := bnds.Max.X * bnds.Max.Y
	rgb[0] = make([]float64, size)
	rgb[1] = make([]float64, size)
	rgb[2] = make([]float64, size)

	var i int
	for y := 0; y < bnds.Max.Y; y++ {
		for x := 0; x < bnds.Max.X; x, i = x+1, i+1 {
			r, g, b, a := img.At(x, y).RGBA()
			r, g, b, a = r>>8, g>>8, b>>8, a>>8
			if a == 255 {
				rgb[0][i] = kSrgbToLinearTable[r]
				rgb[1][i] = kSrgbToLinearTable[g]
				rgb[2][i] = kSrgbToLinearTable[b]
			} else if a == 0 {
				rgb[0][i] = kSrgbToLinearTable[bg]
				rgb[1][i] = kSrgbToLinearTable[bg]
				rgb[2][i] = kSrgbToLinearTable[bg]
			} else {
				hasAlpha = true
				var fg_weight uint = uint(a)
				rgb[0][i] = kSrgbToLinearTable[(uint(r)*fg_weight+127)/255]
				rgb[1][i] = kSrgbToLinearTable[(uint(g)*fg_weight+127)/255]
				rgb[2][i] = kSrgbToLinearTable[(uint(b)*fg_weight+127)/255]
			}
		}
	}

	return rgb[:], hasAlpha
}

func ImageToLinearOnWhite(img image.Image) [][]float64 {

	const bg uint = 255

	var (
		rgb [3][]float64

		bnds = img.Bounds()
	)

	size := bnds.Max.X * bnds.Max.Y
	rgb[0] = make([]float64, size)
	rgb[1] = make([]float64, size)
	rgb[2] = make([]float64, size)

	var i int
	for y := 0; y < bnds.Max.Y; y++ {
		for x := 0; x < bnds.Max.X; x, i = x+1, i+1 {
			r, g, b, a := img.At(x, y).RGBA()
			r, g, b, a = r>>8, g>>8, b>>8, a>>8
			if a == 255 {
				rgb[0][i] = kSrgbToLinearTable[r]
				rgb[1][i] = kSrgbToLinearTable[g]
				rgb[2][i] = kSrgbToLinearTable[b]
			} else if a == 0 {
				rgb[0][i] = kSrgbToLinearTable[bg]
				rgb[1][i] = kSrgbToLinearTable[bg]
				rgb[2][i] = kSrgbToLinearTable[bg]
			} else {
				var fg_weight uint = uint(a)
				var bg_weight uint = 255 - fg_weight
				var bgw uint = (bg * bg_weight) + 127
				rgb[0][i] = kSrgbToLinearTable[(uint(r)*fg_weight+bgw)/255]
				rgb[1][i] = kSrgbToLinearTable[(uint(g)*fg_weight+bgw)/255]
				rgb[2][i] = kSrgbToLinearTable[(uint(b)*fg_weight+bgw)/255]
			}
		}
	}

	return rgb[:]
}

func CompareImages(img1, img2 image.Image) (float64, error) {

	size1 := img1.Bounds().Size()
	size2 := img2.Bounds().Size()

	if size1.X != size2.X || size1.Y != size2.Y {
		return 0, ErrDifferentSizes
	}

	linear1, hasAlpha1 := ImageToLinearOnBlack(img1)
	linear2, hasAlpha2 := ImageToLinearOnBlack(img2)

	var diffmap []float64

	diffValue, err := ButteraugliInterface(size1.X, size1.Y, linear1, linear2, diffmap)
	if err != nil {
		return 0, err
	}

	if hasAlpha1 || hasAlpha2 {

		linear1 = ImageToLinearOnWhite(img1)
		linear2 = ImageToLinearOnWhite(img2)

		diffValueOnWhite, err := ButteraugliInterface(size1.X, size1.Y, linear1, linear2, diffmap)
		if err != nil {
			return 0, err
		}

		if diffValueOnWhite > diffValue {
			diffValue = diffValueOnWhite
		}

	}

	return diffValue, nil

}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func copyVfloat64(a []float64) []float64 {
	b := make([]float64, len(a))
	copy(b, a)
	return b
}

func copyVVfloat64(a [][]float64) [][]float64 {
	b := make([][]float64, len(a))
	for i, v := range a {
		if len(v) > 0 {
			vv := make([]float64, len(v))
			copy(vv, v)
			b[i] = vv
		}
	}
	return b
}

func resizeVfloat64(a *[]float64, size int) {
	if len(*a) < size {
		tmp := make([]float64, size)
		copy(tmp, *a)
		*a = tmp
	} else {
		*a = (*a)[:size]
	}
}

func resizeVVfloat64(a *[][]float64, size int) {
	if len(*a) < size {
		tmp := make([][]float64, size)
		copy(tmp, *a)
		*a = tmp
	} else {
		*a = (*a)[:size]
	}
}
