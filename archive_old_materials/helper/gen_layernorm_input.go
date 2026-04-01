package helper

import (
	"math"

	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// GenerateTestValuesWithVarianceColumnMajor generates test values with proper variance across groups
// Returns values in column-major layout: [pos0_group0, pos0_group1, ..., pos1_group0, ...]
// For each position, generates variation across groups with variance in [minVar, maxVar]
//
// Column-major layout for MaxSlots=2048, hidDim=256, numGroups=8:
//
//	values[0], values[256], values[512], ..., values[1792] = different values for position 0 across 8 groups
//	values[1], values[257], values[513], ..., values[1793] = different values for position 1 across 8 groups
//	etc.
//
// This ensures that when LayerNorm aggregates across groups, it gets proper variance statistics.
func GenerateTestValuesWithVarianceColumnMajor(minVar, maxVar float64, hidDim, numGroups int) []float64 {
	totalSize := hidDim * numGroups
	values := make([]float64, totalSize)

	for g := 0; g < numGroups; g++ {
		// 1. 为每个样本 (Group) 生成初始随机向量
		groupData := make([]float64, hidDim)
		sum := 0.0
		for i := 0; i < hidDim; i++ {
			groupData[i] = sampling.RandFloat64(-1, 1)
			sum += groupData[i]
		}

		// 2. 中心化 (Zero-Mean)
		// LayerNorm 对 Mean Shift 不敏感，但为了精确控制 Variance，我们先去均值
		mean := sum / float64(hidDim)
		varSum := 0.0
		for i := 0; i < hidDim; i++ {
			groupData[i] -= mean
			varSum += groupData[i] * groupData[i]
		}
		currentVar := varSum / float64(hidDim)

		// 3. 随机选择一个目标方差 targetVar ∈ [minVar, maxVar]
		targetVar := minVar + sampling.RandFloat64(0, 1)*(maxVar-minVar)

		// 4. 计算缩放系数，强制调整方差
		// 目标标准差 / 当前标准差
		scale := math.Sqrt(targetVar / currentVar)

		// 5. 填充到全局数组 (应用 Column-Major 布局)
		// 同时加上一个随机 Bias (LayerNorm 应该能自动减去这个 Bias)
		randomBias := sampling.RandFloat64(-2.0, 2.0)

		for i := 0; i < hidDim; i++ {
			// Column-Major Layout:
			// [Group0_Feat0, Group1_Feat0, ..., Group0_Feat1, ...]
			idx := i*numGroups + g

			// 最终值 = (中心化数据 * 缩放系数) + 随机均值
			values[idx] = (groupData[i] * scale) + randomBias
		}
	}
	return values
}
