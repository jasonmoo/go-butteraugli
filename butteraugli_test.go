package butteraugli

import (
	"image"
	"os"
	"testing"

	_ "image/png"
)

func TestCompareImagesLarge(t *testing.T) {

	const (
		image1 = `testdata/up.orig.png`
		image2 = `testdata/up.best.png`
		// compare_pngs: 1.071696
		expected = 1.0716956373428894
	)

	img1, img2 := DecodeImages(t, image1, image2)

	dv, err := CompareImages(img1, img2)
	if err != nil {
		t.Fatal(err)
	}

	if dv != expected {
		t.Errorf("Expected: %f, got %f", expected, dv)
	}

}

func TestCompareImagesMedium(t *testing.T) {

	const (
		image1 = `testdata/up.1024.orig.png`
		image2 = `testdata/up.1024.png`
		// compare_pngs: 1.152866
		expected = 1.1528656298754318
	)

	img1, img2 := DecodeImages(t, image1, image2)

	dv, err := CompareImages(img1, img2)
	if err != nil {
		t.Fatal(err)
	}

	if dv != expected {
		t.Errorf("Expected: %f, got %f", expected, dv)
	}

}

func TestCompareImagesSmall(t *testing.T) {

	const (
		image1 = `testdata/up.400.orig.png`
		image2 = `testdata/up.400.png`
		// compare_pngs: 1.308039
		expected = 1.308038623494388
	)

	img1, img2 := DecodeImages(t, image1, image2)

	dv, err := CompareImages(img1, img2)
	if err != nil {
		t.Fatal(err)
	}

	if dv != expected {
		t.Errorf("Expected: %f, got %f", expected, dv)
	}

}

func DecodeImages(t *testing.T, image1, image2 string) (image.Image, image.Image) {

	file1, err := os.Open(image1)
	if err != nil {
		t.Fatal(err)
	}
	defer file1.Close()

	img1, _, err := image.Decode(file1)
	if err != nil {
		t.Fatal(err)
	}

	file2, err := os.Open(image2)
	if err != nil {
		t.Fatal(err)
	}
	defer file2.Close()

	img2, _, err := image.Decode(file2)
	if err != nil {
		t.Fatal(err)
	}

	return img1, img2

}
