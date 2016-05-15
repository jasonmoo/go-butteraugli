package butteraugli

import (
	"errors"
	"log"
	"math"
	"unsafe"
)

const (
	kButteraugliGood      float64 = 1.000
	kButteraugliBad       float64 = 1.088091
	kButteraugliQuantLow  float64 = 0.26
	kButteraugliQuantHigh float64 = 1.454
)

var (
	ErrTooSmall       = errors.New("Butteraugli is undefined for small images")
	ErrDifferentSizes = errors.New("Images are not the same size")
)

// Allows incremental computation of the butteraugli map by keeping some
// intermediate results.
type ButteraugliComparator struct {
	xsize_      int
	ysize_      int
	num_pixels_ int
	step_       int
	res_xsize_  int
	res_ysize_  int

	// Contains the suppression map, 3 layers, each xsize_ * ysize_ in size.
	scale_xyz_ [][]float64
	// The blurred original used in the edge detector map.
	blurred0_ [][]float64
	// The following are all step_ x step_ subsampled maps containing
	// 3-dimensional vectors.
	gamma_map_         []float64
	dct8x8map_dc_      []float64
	dct8x8map_ac_      []float64
	edge_detector_map_ []float64
}

func NewButteraugliComparator(xsize, ysize, step int) *ButteraugliComparator {
	bc := &ButteraugliComparator{}
	bc.xsize_ = xsize
	bc.ysize_ = ysize
	bc.num_pixels_ = xsize * ysize
	bc.step_ = step
	bc.res_xsize_ = (xsize + step - 1) / step
	bc.res_ysize_ = (ysize + step - 1) / step

	bc.scale_xyz_ = make([][]float64, 3)
	for i, _ := range bc.scale_xyz_ {
		bc.scale_xyz_[i] = make([]float64, bc.num_pixels_)
	}

	size := 3 * bc.res_xsize_ * bc.res_ysize_
	bc.gamma_map_ = make([]float64, size)
	bc.dct8x8map_dc_ = make([]float64, size)
	bc.dct8x8map_ac_ = make([]float64, size)
	bc.edge_detector_map_ = make([]float64, size)
	return bc
}

// Copies the suppression map computed by the previous call to DistanceMap()
// or DistanceMapIncremental() to *suppression.
func (bc *ButteraugliComparator) GetSuppressionMap(suppression *[][]float64) {
	*suppression = bc.scale_xyz_
}

// The high frequency color model used by RgbDiff().
func RgbToXyz(r, g, b float64, valx, valy, valz *float64) {
	const (
		a0 float64 = 0.19334520917582404
		a1 float64 = -0.08311773494921797
		b0 float64 = 0.07713792858953174
		b1 float64 = 0.2208810782725995
		c0 float64 = 0.26188332580170837
	)
	*valx = a0*r + a1*g
	*valy = b0*r + b1*g
	*valz = c0 * b
}

const (
	kInternalGoodQualityThreshold float64 = 12.84
	kGlobalScale                  float64 = 1.0 / kInternalGoodQualityThreshold
)

func DotProduct(u [3]float64, v [3]float64) float64 {
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
}

func DotProductWithMax(u [3]float64, v [3]float64) float64 {
	return (math.Max(u[0], u[0]*v[0]) +
		math.Max(u[1], u[1]*v[1]) +
		math.Max(u[2], u[2]*v[2]))
}

// Computes a horizontal convolution and transposes the result.
func Convolution(xsize, ysize, xstep, length, offset int, multipliers, inp, result []float64) {
	for x, ox := 0, 0; x < xsize; x, ox = x+xstep, ox+1 {
		var (
			minx   int
			maxx   int = min(xsize, x+length-offset) - 1
			weight float64
		)
		if x > offset {
			minx = x - offset
		}
		for j := minx; j <= maxx; j++ {
			weight += multipliers[j-x+offset]
		}
		var scale float64 = 1.0 / weight
		for y := 0; y < ysize; y++ {
			var sum float64
			for j := minx; j <= maxx; j++ {
				sum += inp[y*xsize+j] * multipliers[j-x+offset]
			}
			result[ox*ysize+y] = sum * scale
		}
	}
}

func GaussBlurApproximation(xsize, ysize int, channel []float64, sigma float64) {
	const m float64 = 2.25 // Accuracy increases when m is increased.
	var scaler float64 = -1.0 / (2 * sigma * sigma)
	// For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
	var diff int = max(1, int(m*math.Abs(sigma)))
	var expn_size int = 2*diff + 1
	expn := make([]float64, expn_size)
	for i := -diff; i <= diff; i++ {
		expn[i+diff] = math.Exp(scaler * float64(i*i))
	}
	// No effort was expended to choose good values here.
	var (
		xstep  int = max(1, int(sigma/3))
		ystep  int = xstep
		dxsize int = (xsize + xstep - 1) / xstep
		dysize int = (ysize + ystep - 1) / ystep

		tmp                = make([]float64, dxsize*ysize)
		downsampled_output = make([]float64, dxsize*dysize)
	)
	Convolution(xsize, ysize, xstep, expn_size, diff, expn, channel, tmp)
	Convolution(ysize, dxsize, ystep, expn_size, diff, expn, tmp, downsampled_output)
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			// TODO: Use correct rounding.
			channel[y*xsize+x] = downsampled_output[(y/ystep)*dxsize+(x/xstep)]
		}
	}
}

// Model of the gamma derivative in the human eye.
func GammaDerivativeRaw(v float64) float64 {
	// Derivative of the linear to sRGB translation.
	if v <= 4.1533262364511305 {
		return 6.34239659083478
	}
	return 0.3509337062449116 * math.Pow(v/255.0, -0.7171642149318425)
}

type GammaDerivativeTableEntry struct {
	slope    float64
	constant float64
}

var GammaDerivativeTable = func() []GammaDerivativeTableEntry {
	var kTable [256]GammaDerivativeTableEntry
	var prev float64 = GammaDerivativeRaw(0)
	for i := 0; i < 255; i++ {
		var (
			fi       float64 = float64(i)
			next     float64 = GammaDerivativeRaw(fi + 1)
			slope    float64 = next - prev
			constant float64 = prev - slope*fi
		)
		kTable[i].slope = slope
		kTable[i].constant = constant
		prev = next
	}
	kTable[255].slope = 0.0
	kTable[255].constant = prev
	return kTable[:]
}()

func GammaDerivativeLut(v float64) float64 {
	var i int
	if v < 0 {
		i = 0
	} else if v > 255 {
		i = 255
	} else {
		i = int(v)
	}
	return GammaDerivativeTable[i].slope*v + GammaDerivativeTable[i].constant
}

var csf8x8 = [64]float64{
	0.462845464,
	1.48675033,
	0.774522722,
	0.656786477,
	0.507984559,
	0.51125,
	0.51125,
	0.55125,
	1.48675033,
	0.893383342,
	0.729597657,
	0.644616012,
	0.47125,
	0.47125,
	0.53125,
	0.53125,
	0.774522722,
	0.729597657,
	0.669401271,
	0.547687084,
	0.47125,
	0.47125,
	0.53125,
	0.53125,
	0.656786477,
	0.644616012,
	0.547687084,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.507984559,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.51125,
	0.47125,
	0.47125,
	0.47125,
	0.47125,
	0.53125,
	0.53125,
	0.51125,
	0.51125,
	0.53125,
	0.53125,
	0.47125,
	0.47125,
	0.53125,
	0.47125,
	0.47125,
	0.55125,
	0.53125,
	0.53125,
	0.47125,
	0.47125,
	0.51125,
	0.47125,
	0.51125,
}

// Contrast sensitivity related weights.
func GetContrastSensitivityMatrix() []float64 {
	return csf8x8[:]
}

func Transpose8x8(data *[64]float64) {
	for i := 0; i < 8; i++ {
		for j := 0; j < i; j++ {
			a, b := 8*i+j, 8*j+i
			data[a], data[b] = data[b], data[a]
		}
	}
}

// Perform a DCT on each of the 8 columns. Results is scaled.
// The Dct computation used in butteraugli.
func ButteraugliDctd8x8Vertical(data *[64]float64) {
	const STRIDE int = 8
	for col := 0; col < 8; col++ {
		var (
			dataptr = data[col:]

			tmp0 float64 = dataptr[STRIDE*0] + dataptr[STRIDE*7]
			tmp7 float64 = dataptr[STRIDE*0] - dataptr[STRIDE*7]
			tmp1 float64 = dataptr[STRIDE*1] + dataptr[STRIDE*6]
			tmp6 float64 = dataptr[STRIDE*1] - dataptr[STRIDE*6]
			tmp2 float64 = dataptr[STRIDE*2] + dataptr[STRIDE*5]
			tmp5 float64 = dataptr[STRIDE*2] - dataptr[STRIDE*5]
			tmp3 float64 = dataptr[STRIDE*3] + dataptr[STRIDE*4]
			tmp4 float64 = dataptr[STRIDE*3] - dataptr[STRIDE*4]

			/* Even part */
			tmp10 float64 = tmp0 + tmp3 /* phase 2 */
			tmp13 float64 = tmp0 - tmp3
			tmp11 float64 = tmp1 + tmp2
			tmp12 float64 = tmp1 - tmp2
		)
		dataptr[STRIDE*0] = tmp10 + tmp11 /* phase 3 */
		dataptr[STRIDE*4] = tmp10 - tmp11

		var z1 float64 = (tmp12 + tmp13) * 0.7071067811865476 /* c4 */
		dataptr[STRIDE*2] = tmp13 + z1                        /* phase 5 */
		dataptr[STRIDE*6] = tmp13 - z1

		/* Odd part */
		tmp10 = tmp4 + tmp5 /* phase 2 */
		tmp11 = tmp5 + tmp6
		tmp12 = tmp6 + tmp7

		var (
			z5 float64 = (tmp10 - tmp12) * 0.38268343236508984 /* c6 */
			z2 float64 = 0.5411961001461969*tmp10 + z5         /* c2-c6 */
			z4 float64 = 1.3065629648763766*tmp12 + z5         /* c2+c6 */
			z3 float64 = tmp11 * 0.7071067811865476            /* c4 */

			z11 float64 = tmp7 + z3 /* phase 5 */
			z13 float64 = tmp7 - z3
		)

		dataptr[STRIDE*5] = z13 + z2 /* phase 6 */
		dataptr[STRIDE*3] = z13 - z2
		dataptr[STRIDE*1] = z11 + z4
		dataptr[STRIDE*7] = z11 - z4
	}
}

var kScalingFactors = [64]float64{
	0.0156250000000000, 0.0079655559235025,
	0.0084561890647843, 0.0093960138583601,
	0.0110485434560398, 0.0140621284865065,
	0.0204150463261934, 0.0400455538709610,
	0.0079655559235025, 0.0040608051949085,
	0.0043109278012960, 0.0047900463261934,
	0.0056324986094293, 0.0071688109352157,
	0.0104075003643000, 0.0204150463261934,
	0.0084561890647843, 0.0043109278012960,
	0.0045764565439602, 0.0050850860570641,
	0.0059794286307045, 0.0076103690966521,
	0.0110485434560398, 0.0216724975831585,
	0.0093960138583601, 0.0047900463261934,
	0.0050850860570641, 0.0056502448912957,
	0.0066439851153692, 0.0084561890647843,
	0.0122764837247985, 0.0240811890647843,
	0.0110485434560398, 0.0056324986094293,
	0.0059794286307045, 0.0066439851153692,
	0.0078125000000000, 0.0099434264107253,
	0.0144356176954889, 0.0283164826985277,
	0.0140621284865065, 0.0071688109352157,
	0.0076103690966521, 0.0084561890647843,
	0.0099434264107253, 0.0126555812845451,
	0.0183730562878025, 0.0360400463261934,
	0.0204150463261934, 0.0104075003643000,
	0.0110485434560398, 0.0122764837247985,
	0.0144356176954889, 0.0183730562878025,
	0.0266735434560398, 0.0523220375957595,
	0.0400455538709610, 0.0204150463261934,
	0.0216724975831585, 0.0240811890647843,
	0.0283164826985277, 0.0360400463261934,
	0.0523220375957595, 0.1026333686292507,
}

// The Dct computation used in butteraugli.
func ButteraugliDctd8x8(m *[64]float64) {
	ButteraugliDctd8x8Vertical(m)
	Transpose8x8(m)
	ButteraugliDctd8x8Vertical(m)
	for i := 0; i < 64; i++ {
		m[i] *= kScalingFactors[i]
	}
}

// Mix a little bit of neighbouring pixels into the corners.
func ButteraugliMixCorners(m *[64]float64) {
	const (
		c float64 = 5.772211696386835
		w float64 = 1.0 / (c + 2)
	)
	m[0] = (c*m[0] + m[1] + m[8]) * w
	m[7] = (c*m[7] + m[6] + m[15]) * w
	m[56] = (c*m[56] + m[57] + m[48]) * w
	m[63] = (c*m[63] + m[55] + m[62]) * w
}

// Rgbdiff for one 8x8 block.
// The Dct computation used in butteraugli.
// Computes 8x8 DCT (with corner mixing) of each channel of rgb0 and rgb1 and
// adds the total squared 3-dimensional rgbdiff of the two blocks to diff_xyz.
// making copies of rgb0, rgb1
func ButteraugliDctd8x8RgbDiff(gamma *[3]float64, rgb0, rgb1 [192]float64, diff_xyz_dc, diff_xyz_ac *[3]float64) {
	for c := 0; c < 3; c++ {
		ButteraugliMixCorners((*[64]float64)(unsafe.Pointer(&rgb0[c*64])))
		ButteraugliDctd8x8((*[64]float64)(unsafe.Pointer(&rgb0[c*64])))
		ButteraugliMixCorners((*[64]float64)(unsafe.Pointer(&rgb1[c*64])))
		ButteraugliDctd8x8((*[64]float64)(unsafe.Pointer(&rgb1[c*64])))
	}
	var (
		r0 = rgb0[0:]
		g0 = rgb0[64:]
		b0 = rgb0[2*64:]
		r1 = rgb1[0:]
		g1 = rgb1[64:]
		b1 = rgb1[2*64:]

		rmul float64 = gamma[0]
		gmul float64 = gamma[1]
		bmul float64 = gamma[2]
	)
	RgbDiffLowFreqSquaredXyzAccumulate(rmul*(r0[0]-r1[0]),
		gmul*(g0[0]-g1[0]),
		bmul*(b0[0]-b1[0]),
		0, 0, 0, csf8x8[0]*csf8x8[0],
		diff_xyz_dc)
	for i := 1; i < 64; i++ {
		var d float64 = csf8x8[i] * csf8x8[i]
		RgbDiffSquaredXyzAccumulate(
			rmul*r0[i], gmul*g0[i], bmul*b0[i],
			rmul*r1[i], gmul*g1[i], bmul*b1[i],
			d, diff_xyz_ac)
	}
}

// Direct model with low frequency edge detectors.
// Two edge detectors are applied in each corner of the 8x8 square.
// The squared 3-dimensional error vector is added to diff_xyz.
func Butteraugli8x8CornerEdgeDetectorDiff(pos_x, pos_y, xsize, ysize int, blurred0, blurred1 [][]float64, gamma, diff_xyz *[3]float64) {
	const (
		w      float64 = 0.9375862313610259
		weight float64 = 0.04051114418675643
		step   int     = 3
	)
	var (
		gamma_scaled = [3]float64{gamma[0] * w, gamma[1] * w, gamma[2] * w}
		offset       = [4][2]int{{0, 0}, {0, 7}, {7, 0}, {7, 7}}
	)
	for k := 0; k < 4; k++ {
		var (
			x int = pos_x + offset[k][0]
			y int = pos_y + offset[k][1]
		)
		if x >= step && x+step < xsize {
			var ix int = y*xsize + (x - step)
			var ix2 int = ix + 2*step
			RgbDiffLowFreqSquaredXyzAccumulate(
				gamma_scaled[0]*(blurred0[0][ix]-blurred0[0][ix2]),
				gamma_scaled[1]*(blurred0[1][ix]-blurred0[1][ix2]),
				gamma_scaled[2]*(blurred0[2][ix]-blurred0[2][ix2]),
				gamma_scaled[0]*(blurred1[0][ix]-blurred1[0][ix2]),
				gamma_scaled[1]*(blurred1[1][ix]-blurred1[1][ix2]),
				gamma_scaled[2]*(blurred1[2][ix]-blurred1[2][ix2]),
				weight, diff_xyz)
		}
		if y >= step && y+step < ysize {
			var ix int = (y-step)*xsize + x
			var ix2 int = ix + 2*step*xsize
			RgbDiffLowFreqSquaredXyzAccumulate(
				gamma_scaled[0]*(blurred0[0][ix]-blurred0[0][ix2]),
				gamma_scaled[1]*(blurred0[1][ix]-blurred0[1][ix2]),
				gamma_scaled[2]*(blurred0[2][ix]-blurred0[2][ix2]),
				gamma_scaled[0]*(blurred1[0][ix]-blurred1[0][ix2]),
				gamma_scaled[1]*(blurred1[1][ix]-blurred1[1][ix2]),
				gamma_scaled[2]*(blurred1[2][ix]-blurred1[2][ix2]),
				weight, diff_xyz)
		}
	}
}

func GammaDerivativeWeightedAvg(m0, m1 *[64]float64) float64 {
	const slack float64 = 32
	var (
		total       float64
		total_level float64
	)
	for i := 0; i < 64; i++ {
		var level float64 = math.Min(m0[i], m1[i])
		var w float64 = GammaDerivativeLut(level) * math.Abs(m0[i]-m1[i])
		w += slack
		total += w
		total_level += w * level
	}
	return GammaDerivativeLut(total_level / total)
}

func MixGamma(gamma *[3]float64) {
	var (
		gamma_r float64 = gamma[0]
		gamma_g float64 = gamma[1]
		gamma_b float64 = gamma[2]
	)
	{
		const a1 float64 = 0.031357286184508865
		const a0 float64 = 1.0700981209942968 - a1
		gamma[0] = math.Pow(gamma_r, a0) * math.Pow(gamma_g, a1)
	}
	{
		const a0 float64 = 0.022412459763909404
		const a1 float64 = 1.0444144247574045 - a0
		gamma[1] = math.Pow(gamma_r, a0) * math.Pow(gamma_g, a1)
	}
	{
		const a0 float64 = 0.021865296182264373
		const a1 float64 = 0.040708035758485354
		const a2 float64 = 0.9595217356556563 - a0 - a1
		gamma[2] = math.Pow(gamma_r, a0) * math.Pow(gamma_g, a1) * math.Pow(gamma_b, a2)
	}
}

// Computes the butteraugli map from scratch, updates all intermediate
// results.
func (bc ButteraugliComparator) DistanceMap(rgb0, rgb1 [][]float64, result *[]float64) {
	changed := make([]bool, bc.res_xsize_*bc.res_ysize_)
	for i, _ := range changed {
		changed[i] = true
	}
	bc.blurred0_ = nil
	bc.DistanceMapIncremental(rgb0, rgb1, changed, result)
}

// Computes the butteraugli map by resuing some intermediate results from the
// previous run.
//
// Must be called with the same rgb0 image as in the last DistanceMap() call.
//
// If changed[res_y * res_xsize_ + res_x] is false, it assumes that rgb1
// did not change compared to the previous calls of this function or
// of DistanceMap() anywhere within an 8x8 block with upper-left corner in
// (step_ * res_x, step_ * res_y).
func (bc *ButteraugliComparator) DistanceMapIncremental(rgb0, rgb1 [][]float64, changed []bool, result *[]float64) {
	if !(8 <= bc.xsize_) {
		log.Panicf("8 <= bc.xsize_: %d", bc.xsize_)
	}
	for i := 0; i < 3; i++ {
		if !(len(rgb0[i]) == bc.num_pixels_) {
			log.Panicf("len(rgb0[i]) == bc.num_pixels_: %d : %d", len(rgb0[i]), bc.num_pixels_)
		}
		if !(len(rgb1[i]) == bc.num_pixels_) {
			log.Panicf("len(rgb1[i]) == bc.num_pixels_: %d : %d", len(rgb1[i]), bc.num_pixels_)
		}
	}
	bc.Dct8x8mapIncremental(rgb0, rgb1, changed)
	bc.EdgeDetectorMap(rgb0, rgb1)
	bc.SuppressionMap(rgb0, rgb1)
	bc.FinalizeDistanceMap(result)
}

func (bc *ButteraugliComparator) Dct8x8mapIncremental(rgb0, rgb1 [][]float64, changed []bool) {
	for res_y := 0; res_y+7 < bc.ysize_; res_y += bc.step_ {
		for res_x := 0; res_x+7 < bc.xsize_; res_x += bc.step_ {
			var res_ix int = (res_y*bc.res_xsize_ + res_x) / bc.step_
			if !changed[res_ix] {
				continue
			}
			var (
				block0 [3 * 64]float64
				block1 [3 * 64]float64
				gamma  [3]float64
			)
			for i := 0; i < 3; i++ {
				m0 := (*[64]float64)(unsafe.Pointer(&block0[i*64]))
				m1 := (*[64]float64)(unsafe.Pointer(&block1[i*64]))
				for y := 0; y < 8; y++ {
					for x := 0; x < 8; x++ {
						m0[8*y+x] = rgb0[i][(res_y+y)*bc.xsize_+res_x+x]
						m1[8*y+x] = rgb1[i][(res_y+y)*bc.xsize_+res_x+x]
					}
				}
				gamma[i] = GammaDerivativeWeightedAvg(m0, m1)
			}
			MixGamma(&gamma)
			var diff_xyz_dc, diff_xyz_ac [3]float64
			ButteraugliDctd8x8RgbDiff(&gamma, block0, block1, &diff_xyz_dc, &diff_xyz_ac)
			for i := 0; i < 3; i++ {
				bc.gamma_map_[3*res_ix+i] = gamma[i]
				bc.dct8x8map_dc_[3*res_ix+i] = diff_xyz_dc[i]
				bc.dct8x8map_ac_[3*res_ix+i] = diff_xyz_ac[i]
			}
		}
	}
}

var kSigma = [3]float64{
	1.9467950320244936,
	1.9844023344574777,
	0.9443734014996432,
}

func (bc *ButteraugliComparator) EdgeDetectorMap(rgb0, rgb1 [][]float64) {
	if len(bc.blurred0_) == 0 {
		bc.blurred0_ = copyVVfloat64(rgb0)
		for i := 0; i < 3; i++ {
			GaussBlurApproximation(bc.xsize_, bc.ysize_, bc.blurred0_[i], kSigma[i])
		}
	}
	blurred1 := copyVVfloat64(rgb1)
	for i := 0; i < 3; i++ {
		GaussBlurApproximation(bc.xsize_, bc.ysize_, blurred1[i], kSigma[i])
	}
	for res_y := 0; res_y+7 < bc.ysize_; res_y += bc.step_ {
		for res_x := 0; res_x+7 < bc.xsize_; res_x += bc.step_ {
			var res_ix int = (res_y*bc.res_xsize_ + res_x) / bc.step_
			var gamma [3]float64
			for i := 0; i < 3; i++ {
				gamma[i] = bc.gamma_map_[3*res_ix+i]
			}
			var diff_xyz [3]float64
			Butteraugli8x8CornerEdgeDetectorDiff(res_x, res_y, bc.xsize_, bc.ysize_,
				bc.blurred0_, blurred1,
				&gamma, &diff_xyz)
			for i := 0; i < 3; i++ {
				bc.edge_detector_map_[3*res_ix+i] = diff_xyz[i]
			}
		}
	}
}

func (bc *ButteraugliComparator) CombineChannels(result *[]float64) {
	resizeVfloat64(result, bc.res_xsize_*bc.res_ysize_)
	for res_y := 0; res_y+7 < bc.ysize_; res_y += bc.step_ {
		for res_x := 0; res_x+7 < bc.xsize_; res_x += bc.step_ {
			var res_ix int = (res_y*bc.res_xsize_ + res_x) / bc.step_
			var scale [3]float64
			for i := 0; i < 3; i++ {
				scale[i] = bc.scale_xyz_[i][(res_y+3)*bc.xsize_+(res_x+3)]
				scale[i] *= scale[i]
			}
			dct8x8map_dc_p := (*[3]float64)(unsafe.Pointer(&bc.dct8x8map_dc_[3*res_ix]))
			dct8x8map_ac_p := (*[3]float64)(unsafe.Pointer(&bc.dct8x8map_ac_[3*res_ix]))
			edge_detector_map_p := (*[3]float64)(unsafe.Pointer(&bc.edge_detector_map_[3*res_ix]))
			(*result)[res_ix] =
				// Apply less suppression to the low frequency component, otherwise
				// it will not rate a difference on a noisy image where the average
				// color changes visibly (especially for blue) high enough. The "max"
				// formula is chosen ad-hoc, as a balance between fixing the noisy
				// image issue (the ideal fix would mean only multiplying the diff
				// with a constant here), and staying aligned with user ratings on
				// other images: take the highest diff of both options.
				(DotProductWithMax(*dct8x8map_dc_p, scale) +
					DotProduct(*dct8x8map_ac_p, scale) +
					DotProduct(*edge_detector_map_p, scale))
		}
	}
}

func SoftClampHighValues(v float64) float64 {
	if v < 0 {
		return 0
	}
	return math.Log(v + 1.0)
}

// Making a cluster of local errors to be more impactful than
// just a single error.
func ApplyErrorClustering(xsize, ysize, step int, distmap *[]float64) {
	// Upsample and take square root.
	distmap_out := make([]float64, xsize*ysize)
	var res_xsize int = (xsize + step - 1) / step
	for res_y := 0; res_y+7 < ysize; res_y += step {
		for res_x := 0; res_x+7 < xsize; res_x += step {
			var res_ix int = (res_y*res_xsize + res_x) / step
			var val = math.Sqrt((*distmap)[res_ix])
			for off_y := 0; off_y < step; off_y++ {
				for off_x := 0; off_x < step; off_x++ {
					distmap_out[(res_y+off_y)*xsize+res_x+off_x] = val
				}
			}
		}
	}
	*distmap = distmap_out
	{
		const (
			kSigma float64 = 17.77401417750591
			mul1   float64 = 6.127295867565043
			mul2   float64 = 1.5735505590879941
		)
		blurred := copyVfloat64(*distmap)
		GaussBlurApproximation(xsize, ysize, blurred, kSigma)
		for i := 0; i < ysize*xsize; i++ {
			(*distmap)[i] += mul1 * SoftClampHighValues(mul2*blurred[i])
		}
	}
	{
		const (
			kSigma float64 = 11.797090536094919
			mul1   float64 = 2.9304661498619393
			mul2   float64 = 2.5400911895122853
		)
		blurred := copyVfloat64(*distmap)
		GaussBlurApproximation(xsize, ysize, blurred, kSigma)
		for i := 0; i < ysize*xsize; i++ {
			(*distmap)[i] += mul1 * SoftClampHighValues(mul2*blurred[i])
		}
	}
}

func MultiplyScalarImage(xsize, ysize, offset int, scale, result []float64) {
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			var idx int = min(y+offset, ysize-1) * xsize
			idx += min(x+offset, xsize-1)
			var v float64 = scale[idx]
			if !(0 < v) {
				log.Panicf("0 < v: %f", v)
			}
			result[x+y*xsize] *= v
		}
	}
}

func ScaleImage(scale float64, result []float64) {
	for i := 0; i < len(result); i++ {
		result[i] *= scale
	}
}

func ButteraugliMap(xsize, ysize int, rgb0, rgb1 [][]float64, result *[]float64) {
	bc := NewButteraugliComparator(xsize, ysize, 3)
	bc.DistanceMap(rgb0, rgb1, result)
}

func (bc *ButteraugliComparator) SuppressionMap(rgb0, rgb1 [][]float64) {
	var rgb_avg [3][]float64
	for i := 0; i < 3; i++ {
		rgb_avg[i] = make([]float64, bc.num_pixels_)
		for x := 0; x < bc.num_pixels_; x++ {
			rgb_avg[i][x] = (rgb0[i][x] + rgb1[i][x]) * 0.5
		}
	}
	SuppressionRgb(rgb_avg[:], bc.xsize_, bc.ysize_, &bc.scale_xyz_)
}

func (bc *ButteraugliComparator) FinalizeDistanceMap(result *[]float64) {
	bc.CombineChannels(result)
	ApplyErrorClustering(bc.xsize_, bc.ysize_, bc.step_, result)
	ScaleImage(kGlobalScale, *result)
}

func ButteraugliDistanceFromMap(xsize, ysize int, distmap []float64) float64 {
	var retval float64
	for y := 0; y+7 < ysize; y++ {
		for x := 0; x+7 < xsize; x++ {
			var v float64 = distmap[y*xsize+x]
			if v > retval {
				retval = v
			}
		}
	}
	return retval
}

var kHighFrequencyColorDiffDx = [21]float64{
	0,
	0.2907966745057564,
	0.4051314804057564,
	0.4772760965057564,
	0.5051493856057564,
	2.1729859604144055,
	3.3884646055698626,
	4.0229515574578265,
	4.816434992428891,
	4.902122343469863,
	5.340254095828891,
	5.575275366425944,
	6.2865515546259445,
	6.8836782708259445,
	7.441068346525944,
	7.939018098125944,
	8.369172080625944,
	8.985082806466515,
	9.334801499366515,
	9.589734180466515,
	9.810687724466515,
}

func GetHighFreqColorDiffDx() []float64 {
	return kHighFrequencyColorDiffDx[:]
}

var kLowFrequencyColorDiff = [21]float64{
	0,
	1.1,
	1.5876101261422835,
	1.6976101261422836,
	1.7976101261422834,
	1.8476101261422835,
	1.8976101261422835,
	4.397240606610438,
	4.686599568139378,
	6.186599568139378,
	6.486599568139378,
	6.686599568139378,
	6.886599568139378,
	7.086599568139378,
	7.229062896585699,
	7.3290628965856985,
	7.429062896585698,
	7.529062896585699,
	7.629062896585698,
	7.729062896585699,
	7.8290628965856985,
}

func GetLowFreqColorDiff() []float64 {
	return kLowFrequencyColorDiff[:]
}

var kHighFrequencyColorDiffDy = [21]float64{
	0,
	2.711686498564926,
	6.372713948578674,
	6.749065994565258,
	7.206288924241256,
	7.810763383046433,
	9.06633982465914,
	10.247027693354854,
	11.405609673354855,
	12.514884802454853,
	13.598623651254854,
	14.763543981054854,
	15.858015071354854,
	16.940461051054857,
	18.044423211343567,
	19.148278633543566,
	20.262330053243566,
	21.402794191043565,
	22.499732267943568,
	23.613819632843565,
	24.717384172243566,
}

func GetHighFreqColorDiffDy() []float64 {
	return kHighFrequencyColorDiffDy[:]
}

var kHighFrequencyColorDiffDz = [21]float64{
	0,
	0.5238354062,
	0.6113836358,
	0.7872517703,
	0.8472517703,
	0.9772517703,
	1.1072517703,
	1.2372517703,
	1.3672517703,
	1.4972517703,
	1.5919694734,
	1.7005177031,
	1.8374517703,
	2.0024517703,
	2.2024517703,
	2.4024517703,
	2.6024517703,
	2.8654517703,
	3.0654517703,
	3.2654517703,
	3.4654517703,
}

func GetHighFreqColorDiffDz() []float64 {
	return kHighFrequencyColorDiffDz[:]
}

func Interpolate(array []float64, size int, sx float64) float64 {
	var ix float64 = math.Abs(sx)
	if !(ix < 10000) {
		log.Panicf("ix < 10000: %f", ix)
	}
	var baseix int = int(ix)
	var res float64
	if baseix >= size-1 {
		res = array[size-1]
	} else {
		var mix float64 = ix - float64(baseix)
		var nextix int = baseix + 1
		res = array[baseix] + mix*(array[nextix]-array[baseix])
	}
	if sx < 0 {
		res = -res
	}
	return res
}

func XyzToVals(x, y, z float64, valx, valy, valz *float64) {
	const (
		xmul float64 = 0.6111808709773186
		ymul float64 = 0.6254434332781222
		zmul float64 = 1.3392224065403562
	)
	*valx = xmul * Interpolate(GetHighFreqColorDiffDx(), 21, x)
	*valy = ymul * Interpolate(GetHighFreqColorDiffDy(), 21, y)
	*valz = zmul * Interpolate(GetHighFreqColorDiffDz(), 21, z)
}

func RgbToVals(r, g, b float64, valx, valy, valz *float64) {
	var x, y, z float64
	RgbToXyz(r, g, b, &x, &y, &z)
	XyzToVals(x, y, z, valx, valy, valz)
}

// Rough psychovisual distance to gray for low frequency colors.
func RgbLowFreqToVals(r, g, b float64, valx, valy, valz *float64) {
	const (
		mul0 float64 = 1.174114936496674
		mul1 float64 = 1.1447743969198858
		a0   float64 = mul0 * 0.1426666666666667
		a1   float64 = mul0 * -0.065
		b0   float64 = mul1 * 0.10
		b1   float64 = mul1 * 0.12
		b2   float64 = mul1 * 0.023721063454977084
		c0   float64 = 0.14367553580758044
		xmul float64 = 1.0175474206944557
		ymul float64 = 1.0017393502266154
		zmul float64 = 1.0378355409050648
	)
	var (
		x float64 = a0*r + a1*g
		y float64 = b0*r + b1*g + b2*b
		z float64 = c0 * b
	)
	*valx = xmul * Interpolate(GetLowFreqColorDiff(), 21, x)
	*valy = ymul * Interpolate(GetHighFreqColorDiffDy(), 21, y)
	// We use the same table for x and z for the low frequency colors.
	*valz = zmul * Interpolate(GetLowFreqColorDiff(), 21, z)
}

func RgbDiffSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, factor float64, res *[3]float64) {
	var (
		valx0, valy0, valz0,
		valx1, valy1, valz1 float64
	)
	if r0 == r1 && g0 == g1 && b0 == b1 {
		return
	}
	RgbToVals(r0, g0, b0, &valx0, &valy0, &valz0)
	if r1 == 0.0 && g1 == 0.0 && b1 == 0.0 {
		res[0] += factor * valx0 * valx0
		res[1] += factor * valy0 * valy0
		res[2] += factor * valz0 * valz0
		return
	}
	RgbToVals(r1, g1, b1, &valx1, &valy1, &valz1)
	// Approximate the distance of the colors by their respective distances
	// to gray.
	var (
		valx float64 = valx0 - valx1
		valy float64 = valy0 - valy1
		valz float64 = valz0 - valz1
	)
	res[0] += factor * valx * valx
	res[1] += factor * valy * valy
	res[2] += factor * valz * valz
}

// Function to estimate the psychovisual impact of a high frequency difference.
// Same as RgbDiff^2. Avoids unnecessary square root.
func RgbDiffSquared(r0, g0, b0, r1, g1, b1 float64) float64 {
	var vals [3]float64
	RgbDiffSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, &vals)
	return vals[0] + vals[1] + vals[2]
}

// Function to estimate the psychovisual impact of a high frequency difference.
func RgbDiffScaledSquared(r0, g0, b0, r1, g1, b1 float64, scale [3]float64) float64 {
	var vals [3]float64
	RgbDiffSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, &vals)
	return DotProduct(vals, scale)
}

// Color difference evaluation for 'high frequency' color changes.
//
// Color difference is computed between two color pairs:
// (r0, g0, b0) and (r1, g1, b1).
//
// Values are expected to be between 0 and 255, but gamma corrected,
// i.e., a linear amount of photons is modulated with a fixed amount
// of change in these values.
func RgbDiff(r0, g0, b0, r1, g1, b1 float64) float64 {
	return math.Sqrt(RgbDiffSquared(r0, g0, b0, r1, g1, b1))
}

func RgbDiffLowFreqSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, factor float64, res *[3]float64) {
	var (
		valx0, valy0, valz0,
		valx1, valy1, valz1 float64
	)
	RgbLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0)
	if r1 == 0.0 && g1 == 0.0 && b1 == 0.0 {
		res[0] += factor * valx0 * valx0
		res[1] += factor * valy0 * valy0
		res[2] += factor * valz0 * valz0
		return
	}
	RgbLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1)
	// Approximate the distance of the colors by their respective distances
	// to gray.
	var (
		valx float64 = valx0 - valx1
		valy float64 = valy0 - valy1
		valz float64 = valz0 - valz1
	)
	res[0] += factor * valx * valx
	res[1] += factor * valy * valy
	res[2] += factor * valz * valz
}

// Same as RgbDiffLowFreq^2. Avoids unnecessary square root.
func RgbDiffLowFreqSquared(r0, g0, b0, r1, g1, b1 float64) float64 {
	var vals [3]float64
	RgbDiffLowFreqSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, &vals)
	return vals[0] + vals[1] + vals[2]
}

func RgbDiffLowFreqScaledSquared(r0, g0, b0, r1, g1, b1 float64, scale [3]float64) float64 {
	var vals [3]float64
	RgbDiffLowFreqSquaredXyzAccumulate(r0, g0, b0, r1, g1, b1, 1.0, &vals)
	return DotProduct(vals, scale)
}

func RgbDiffLowFreq(r0, g0, b0, r1, g1, b1 float64) float64 {
	return math.Sqrt(RgbDiffLowFreqSquared(r0, g0, b0, r1, g1, b1))
}

// Version of rgb diff that applies gamma correction to the diffs.
// The rgb "background" values where the diffs occur are given as
// ave_r, ave_g, ave_b.
func RgbDiffGamma(ave_r, ave_g, ave_b, r0, g0, b0, r1, g1, b1 float64) float64 {
	var (
		rmul float64 = GammaDerivativeLut(ave_r)
		gmul float64 = GammaDerivativeLut(ave_g)
		bmul float64 = GammaDerivativeLut(ave_b)
	)
	return RgbDiff(r0*rmul, g0*gmul, b0*bmul, r1*rmul, g1*gmul, b1*bmul)
}

func RgbDiffGammaLowFreq(ave_r, ave_g, ave_b, r0, g0, b0, r1, g1, b1 float64) float64 {
	var (
		rmul float64 = GammaDerivativeLut(ave_r)
		gmul float64 = GammaDerivativeLut(ave_g)
		bmul float64 = GammaDerivativeLut(ave_b)
	)
	return RgbDiffLowFreq(r0*rmul, g0*gmul, b0*bmul, r1*rmul, g1*gmul, b1*bmul)
}

// Fills in coeffs[0..191] vector in such a way that if d[0..191] is the
// difference vector of the XYZ-color space DCT coefficients of an 8x8 block,
// then the butteraugli block error can be approximated with the
//   SUM(coeffs[i] * d[i]^2; i in 0..191)
// quadratic function.
func ButteraugliQuadraticBlockDiffCoeffsXyz(scale, gamma *[3]float64, rgb_copy [192]float64, coeffs *[192]float64) {
	for c := 0; c < 3; c++ {
		ButteraugliDctd8x8((*[64]float64)(unsafe.Pointer(&rgb_copy[c*64])))
	}
	for i, _ := range coeffs {
		coeffs[i] = 0
	}
	for i := 1; i < 64; i++ {
		var (
			r float64 = gamma[0] * rgb_copy[i]
			g float64 = gamma[1] * rgb_copy[i+64]
			b float64 = gamma[2] * rgb_copy[i+128]

			x, y, z float64
		)
		RgbToXyz(r, g, b, &x, &y, &z)
		var vals_a, vals_b [3]float64
		XyzToVals(x+0.5, y+0.5, z+0.5, &vals_a[0], &vals_a[1], &vals_a[2])
		XyzToVals(x-0.5, y-0.5, z-0.5, &vals_b[0], &vals_b[1], &vals_b[2])
		var (
			slopex float64 = vals_a[0] - vals_b[0]
			slopey float64 = vals_a[1] - vals_b[1]
			slopez float64 = vals_a[2] - vals_b[2]
			d      float64 = csf8x8[i] * csf8x8[i]
		)
		coeffs[i] = d * slopex * slopex * scale[0]
		coeffs[i+64] = d * slopey * slopey * scale[1]
		coeffs[i+128] = d * slopez * slopez * scale[2]
	}
}

var lut2 = [...]float64{
	2.4080920167439297,
	1.7310517871734234,
	1.6530012641923442,
	1.2793750898946559,
	1.174310066587132,
	1.08674227374109,
	0.9395922737410899,
	0.74560027001709,
	0.625061913513,
	0.585615549987,
	0.545015549987,
	0.523015549987,
	0.523015549987,
}

// Non-linearities for component-based suppression.
func SuppressionRedPlusGreen(delta float64) float64 {
	return Interpolate(lut2[:], len(lut2), delta)
}

var lut3 = [...]float64{
	3.439854711245826,
	0.9057815912437459,
	0.8795942436511097,
	0.8691824600776474,
	0.8591824600776476,
	0.847448339843269,
	0.82587,
	0.80442724547356859,
	0.69762724547356858,
	0.66102724547356861,
	0.59512224547356851,
	0.57312224547356849,
	0.55312224547356847,
	0.52912224547356856,
	0.50512224547356854,
	0.50512224547356854,
}

func SuppressionRedMinusGreen(delta float64) float64 {
	return Interpolate(lut3[:], len(lut3), delta)
}

var lut4 = []float64{
	1.796130974060199,
	1.7586413079453862,
	1.7268200670818195,
	1.6193338330644527,
	1.4578801627801556,
	1.05,
	0.95,
	0.8963665327,
	0.7844709485,
	0.71381616456428487,
	0.67745725036428484,
	0.64597852966428482,
	0.63454542736428488,
	0.6257514661642849,
	0.59191965086428489,
	0.56379229756428484,
	0.53215685696428483,
	0.50415685696428492,
	0.50415685696428492,
}

func SuppressionBlue(delta float64) float64 {
	return Interpolate(lut4, len(lut4), delta)
}

// Replaces values[x + y * xsize] with the minimum of the values in the
// square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
func MinSquareVal(square_size, offset, xsize, ysize int, values []float64) {
	// offset is not negative and smaller than square_size.
	if !(offset < square_size) {
		log.Panicf("offset < square_size: %d : %d", offset, square_size)
	}
	tmp := make([]float64, xsize*ysize)
	for y := 0; y < ysize; y++ {
		var (
			minh int
			maxh int = min(ysize, y+square_size-offset)
		)
		if offset < y {
			minh = y - offset
		}
		for x := 0; x < xsize; x++ {
			var min float64 = values[x+minh*xsize]
			for j := minh + 1; j < maxh; j++ {
				if v := values[x+j*xsize]; v < min {
					min = v
				}
			}
			tmp[x+y*xsize] = min
		}
	}
	for x := 0; x < xsize; x++ {
		var (
			minw int
			maxw int = min(xsize, x+square_size-offset)
		)
		if offset < x {
			minw = x - offset
		}
		for y := 0; y < ysize; y++ {
			var min float64 = tmp[minw+y*xsize]
			for j := minw + 1; j < maxw; j++ {
				if v := tmp[j+y*xsize]; v < min {
					min = v
				}
			}
			values[x+y*xsize] = min
		}
	}
}

// assert kRadialWeightSize%2 == 1
const kRadialWeightSize int = 5

func RadialWeights(output []float64) float64 {
	const (
		limit float64 = 1.946968773063937
		rnge  float64 = 0.28838147488925875
	)
	var total_weight float64
	for i := 0; i < kRadialWeightSize; i++ {
		for j := 0; j < kRadialWeightSize; j++ {
			var (
				ddx int     = i - kRadialWeightSize/2
				ddy int     = j - kRadialWeightSize/2
				d   float64 = math.Sqrt(float64(ddx*ddx + ddy*ddy))

				result float64
			)
			if d < limit {
				result = 1.0
			} else if d < limit+rnge {
				result = 1.0 - (d-limit)*(1.0/rnge)
			} else {
				result = 0
			}
			output[i*kRadialWeightSize+j] = result
			total_weight += result
		}
	}
	return total_weight
}

// ===== Functions used by Suppression() only =====
func Average5x5(xsize, ysize int, diffs []float64) {
	tmp := copyVfloat64(diffs)
	var (
		patch            [kRadialWeightSize * kRadialWeightSize]float64
		total_weight     float64 = RadialWeights(patch[:])
		total_weight_inv float64 = 1.0 / total_weight
	)
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			var sum float64
			for dy := 0; dy < kRadialWeightSize; dy++ {
				var yy int = y - kRadialWeightSize/2 + dy
				if yy < 0 || yy >= ysize {
					continue
				}
				var dx int
				var xlim int = min(xsize, x+kRadialWeightSize/2+1)
				for xx := max(0, x-kRadialWeightSize/2); xx < xlim; xx, dx = xx+1, dx+1 {
					var ix int = yy*xsize + xx
					var w float64 = patch[dy*kRadialWeightSize+dx]
					sum += w * tmp[ix]
				}
			}
			diffs[y*xsize+x] = sum * total_weight_inv
		}
	}
}

var (
	kSigma2 = [3]float64{
		1.5406666666666667,
		1.5745555555555555,
		0.7178888888888888,
	}
	muls = [3]float64{
		0.9594825346868103,
		0.9594825346868103,
		0.563781684306615,
	}
)

func DiffPrecompute(rgb [][]float64, xsize, ysize int, suppression *[][]float64) {
	var size int = xsize * ysize
	*suppression = copyVVfloat64(rgb)
	for i := 0; i < 3; i++ {
		GaussBlurApproximation(xsize, ysize, (*suppression)[i], kSigma2[i])
		for x := 0; x < size; x++ {
			(*suppression)[i][x] = muls[i] * (rgb[i][x] + (*suppression)[i][x])
		}
	}
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			var (
				ix int = x + xsize*y

				valsh, valsv [3]float64
			)
			if x+1 < xsize {
				var (
					ix2   int     = ix + 1
					ave_r float64 = ((*suppression)[0][ix] + (*suppression)[0][ix2]) * 0.5
					ave_g float64 = ((*suppression)[1][ix] + (*suppression)[1][ix2]) * 0.5
					ave_b float64 = ((*suppression)[2][ix] + (*suppression)[2][ix2]) * 0.5
					gamma         = [3]float64{
						GammaDerivativeLut(ave_r),
						GammaDerivativeLut(ave_g),
						GammaDerivativeLut(ave_b),
					}
				)
				MixGamma(&gamma)
				var (
					r0 float64 = gamma[0] * ((*suppression)[0][ix] - (*suppression)[0][ix2])
					g0 float64 = gamma[1] * ((*suppression)[1][ix] - (*suppression)[1][ix2])
					b0 float64 = gamma[2] * ((*suppression)[2][ix] - (*suppression)[2][ix2])
				)
				RgbToVals(r0, g0, b0, &valsh[0], &valsh[1], &valsh[2])
			}
			if y+1 < ysize {
				var (
					ix2   int     = ix + xsize
					ave_r float64 = ((*suppression)[0][ix] + (*suppression)[0][ix2]) * 0.5
					ave_g float64 = ((*suppression)[1][ix] + (*suppression)[1][ix2]) * 0.5
					ave_b float64 = ((*suppression)[2][ix] + (*suppression)[2][ix2]) * 0.5
					gamma         = [3]float64{
						GammaDerivativeLut(ave_r),
						GammaDerivativeLut(ave_g),
						GammaDerivativeLut(ave_b),
					}
				)
				MixGamma(&gamma)
				var (
					r0 float64 = gamma[0] * ((*suppression)[0][ix] - (*suppression)[0][ix2])
					g0 float64 = gamma[1] * ((*suppression)[1][ix] - (*suppression)[1][ix2])
					b0 float64 = gamma[2] * ((*suppression)[2][ix] - (*suppression)[2][ix2])
				)
				RgbToVals(r0, g0, b0, &valsv[0], &valsv[1], &valsv[2])
			}
			for i := 0; i < 3; i++ {
				(*suppression)[i][ix] = 0.5 * (math.Abs(valsh[i]) + math.Abs(valsv[i]))
			}
		}
	}
}

var (
	sigma = [3]float64{
		13.963188902126857,
		14.912114324178102,
		12.316604481444129,
	}
	muls2 = [3]float64{
		34.702156451753055,
		4.259296809697752,
		30.51708015595755,
	}
)

// Compute values of local frequency masking based on the activity
// in the argb image.
func SuppressionRgb(rgb [][]float64, xsize, ysize int, suppression *[][]float64) {
	DiffPrecompute(rgb, xsize, ysize, suppression)
	for i := 0; i < 3; i++ {
		Average5x5(xsize, ysize, (*suppression)[i])
		MinSquareVal(14, 3, xsize, ysize, (*suppression)[i])
		GaussBlurApproximation(xsize, ysize, (*suppression)[i], sigma[i])
	}
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			const mix float64 = 0.003513839880391094
			var (
				idx int     = y*xsize + x
				a   float64 = (*suppression)[0][idx]
				b   float64 = (*suppression)[1][idx]
				c   float64 = (*suppression)[2][idx]
			)
			(*suppression)[0][idx] = SuppressionRedMinusGreen(muls2[0]*a + mix*b)
			(*suppression)[1][idx] = SuppressionRedPlusGreen(muls2[1] * b)
			(*suppression)[2][idx] = SuppressionBlue(muls2[2]*c + mix*b)
		}
	}
}

func ButteraugliInterface(xsize, ysize int, rgb0, rgb1 [][]float64, diffmap []float64) (float64, error) {

	if xsize < 32 || ysize < 32 {
		return 0, ErrTooSmall
	}

	var size int = xsize * ysize

	for i := 0; i < 3; i++ {
		if len(rgb0[i]) != size || len(rgb1[i]) != size {
			return 0, ErrDifferentSizes
		}
	}

	ButteraugliMap(xsize, ysize, rgb0, rgb1, &diffmap)

	return ButteraugliDistanceFromMap(xsize, ysize, diffmap), nil

}

// Returns a map which can be used for adaptive quantization. Values can
// typically range from kButteraugliQuantLow to kButteraugliQuantHigh. Low
// values require coarse quantization (e.g. near random noise), high values
// require fine quantization (e.g. in smooth bright areas).
func ButteraugliAdaptiveQuantization(xsize, ysize int, rgb [][]float64, quant *[]float64) bool {
	if xsize < 32 || ysize < 32 {
		return false // Butteraugli is undefined for small images.
	}
	var size int = xsize * ysize
	scale_xyz := make([][]float64, 3)
	SuppressionRgb(rgb, xsize, ysize, &scale_xyz)
	resizeVfloat64(quant, size)

	// Multiply the result of suppression and intensity masking together.
	// Suppression gives us values in 3 color channels, but for now we take only
	// the intensity channel.
	for i := 0; i < size; i++ {
		(*quant)[i] = scale_xyz[1][i]
	}
	return true
}

// Converts the butteraugli score into fuzzy class values that are continuous
// at the class boundary. The class boundary location is based on human
// raters, but the slope is arbitrary. Particularly, it does not reflect
// the expectation value of probabilities of the human raters. It is just
// expected that a smoother class boundary will allow for higher-level
// optimization algorithms to work faster.
//
// Returns 2.0 for a perfect match, and 1.0 for 'ok', 0.0 for bad. Because the
// scoring is fuzzy, a butteraugli score of 0.96 would return a class of
// around 1.9.
func ButteraugliFuzzyClass(score float64) float64 {
	// Interesting values of fuzzy_width range from 10 to 1000. The smaller the
	// value, the smoother the class boundaries, and more images will
	// participate in a higher level optimization.
	const (
		fuzzy_width float64 = 55
		fuzzy_good  float64 = fuzzy_width / kButteraugliGood
		fuzzy_ok    float64 = fuzzy_width / kButteraugliBad
	)
	var (
		good_class float64 = 1.0 / (1.0 + math.Exp((score-kButteraugliGood)*fuzzy_good))
		ok_class   float64 = 1.0 / (1.0 + math.Exp((score-kButteraugliBad)*fuzzy_ok))
	)
	return ok_class + good_class
}
