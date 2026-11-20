import { MAX_QUERY_WORDS, MAX_WORD_FEATURES } from './features'

/** A query has one unit for the whole text and one per word */
export const MAX_UNITS = MAX_QUERY_WORDS + 1

/** Each word occurs in the unit of the whole query and in its own unit */
const MAX_UNIT_WORDS = MAX_QUERY_WORDS * 2

/** Best words that the GPU keeps per unit */
export const TOP_WORDS = 24

/** Workgroup width of the scoring pass */
export const SCORE_GROUP = 64

/**
 * Layout of the input buffer of one call, in 32-bit unsigned integers:
 * the unit count, then for each unit the index of its first word,
 * then for each word the index of its first feature, then the feature rows.
 */
export const INPUT_LAYOUT = {
  unitStart: 1,
  wordStart: 1 + MAX_UNITS + 1,
  featureRows: 1 + MAX_UNITS + 1 + MAX_UNIT_WORDS + 1,
  length: 1 + MAX_UNITS + 1 + MAX_UNIT_WORDS + 1 + MAX_UNIT_WORDS * MAX_WORD_FEATURES,
}

export interface ShaderShape {
  dim: number
  hidden: number
  /** Number of words in the built-in word list */
  words: number
}

/** Offsets of the small tensors in the `params` buffer, in floats */
export function paramLayout({ dim, hidden }: ShaderShape): Record<'gateWeight' | 'gateBias' | 'fc1Weight' | 'fc1Bias' | 'fc2Weight' | 'fc2Bias' | 'length', number> {
  const gateBias = dim
  const fc1Weight = gateBias + 1
  const fc1Bias = fc1Weight + hidden * dim
  const fc2Weight = fc1Bias + hidden
  const fc2Bias = fc2Weight + dim * hidden
  return {
    gateWeight: 0,
    gateBias,
    fc1Weight,
    fc1Bias,
    fc2Weight,
    fc2Bias,
    length: fc2Bias + dim,
  }
}

/**
 * WGSL source of the three compute passes, specialized for one model shape.
 *
 * - `encode`: one invocation per unit turns feature rows into a unit vector.
 * - `score`: one invocation per unit and word computes a cosine.
 * - `select`: one invocation per unit keeps the words with the highest cosines.
 */
export function createShader(shape: ShaderShape): string {
  const params = paramLayout(shape)
  return /* wgsl */`
const DIM = ${shape.dim}u;
const HIDDEN = ${shape.hidden}u;
const WORDS = ${shape.words}u;
const TOP = ${TOP_WORDS}u;
const UNIT_START = ${INPUT_LAYOUT.unitStart}u;
const WORD_START = ${INPUT_LAYOUT.wordStart}u;
const FEATURE_ROWS = ${INPUT_LAYOUT.featureRows}u;
const GATE_BIAS = ${params.gateBias}u;
const FC1_WEIGHT = ${params.fc1Weight}u;
const FC1_BIAS = ${params.fc1Bias}u;
const FC2_WEIGHT = ${params.fc2Weight}u;
const FC2_BIAS = ${params.fc2Bias}u;

struct Best {
  word: u32,
  score: f32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
@group(0) @binding(3) var<storage, read_write> unitsOut: array<f32>;

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) id: vec3u) {
  let unit = id.x;
  if (unit >= input[0]) {
    return;
  }
  var pooled = array<f32, DIM>();
  var total = 0.0;
  for (var word = input[UNIT_START + unit]; word < input[UNIT_START + unit + 1u]; word++) {
    let first = input[WORD_START + word];
    let last = input[WORD_START + word + 1u];
    // The explicit zero value matters: without it, the vector of the previous word leaks into this one
    var vector = array<f32, DIM>();
    for (var feature = first; feature < last; feature++) {
      let row = input[FEATURE_ROWS + feature] * DIM;
      for (var i = 0u; i < DIM; i++) {
        vector[i] += features[row + i];
      }
    }
    let scale = 1.0 / f32(last - first);
    var gate = params[GATE_BIAS];
    for (var i = 0u; i < DIM; i++) {
      vector[i] *= scale;
      gate += vector[i] * params[i];
    }
    gate = 1.0 / (1.0 + exp(-gate));
    total += gate;
    for (var i = 0u; i < DIM; i++) {
      pooled[i] += gate * vector[i];
    }
  }
  for (var i = 0u; i < DIM; i++) {
    pooled[i] /= total + 1e-6;
  }
  var hidden: array<f32, HIDDEN>;
  for (var h = 0u; h < HIDDEN; h++) {
    var sum = params[FC1_BIAS + h];
    for (var i = 0u; i < DIM; i++) {
      sum += pooled[i] * params[FC1_WEIGHT + h * DIM + i];
    }
    hidden[h] = max(sum, 0.0);
  }
  var output: array<f32, DIM>;
  var norm = 0.0;
  for (var i = 0u; i < DIM; i++) {
    var sum = pooled[i] + params[FC2_BIAS + i];
    for (var h = 0u; h < HIDDEN; h++) {
      sum += hidden[h] * params[FC2_WEIGHT + i * HIDDEN + h];
    }
    output[i] = sum;
    norm += sum * sum;
  }
  let length = max(sqrt(norm), 1e-12);
  for (var i = 0u; i < DIM; i++) {
    unitsOut[unit * DIM + i] = output[i] / length;
  }
}

@group(0) @binding(4) var<storage, read> unitsIn: array<f32>;
@group(0) @binding(5) var<storage, read> words: array<f32>;
@group(0) @binding(6) var<storage, read_write> scoresOut: array<f32>;

@compute @workgroup_size(${SCORE_GROUP})
fn score(@builtin(global_invocation_id) id: vec3u) {
  let word = id.x;
  let unit = id.y;
  if (word >= WORDS || unit >= input[0]) {
    return;
  }
  var sum = 0.0;
  for (var i = 0u; i < DIM; i++) {
    sum += unitsIn[unit * DIM + i] * words[word * DIM + i];
  }
  scoresOut[unit * WORDS + word] = sum;
}

@group(0) @binding(7) var<storage, read> scoresIn: array<f32>;
@group(0) @binding(8) var<storage, read_write> best: array<Best>;

@compute @workgroup_size(1)
fn select(@builtin(global_invocation_id) id: vec3u) {
  let unit = id.x;
  if (unit >= input[0]) {
    return;
  }
  var topWord: array<u32, TOP>;
  var topScore: array<f32, TOP>;
  for (var i = 0u; i < TOP; i++) {
    topScore[i] = -2.0;
  }
  for (var word = 0u; word < WORDS; word++) {
    let value = scoresIn[unit * WORDS + word];
    if (value <= topScore[TOP - 1u]) {
      continue;
    }
    var slot = TOP - 1u;
    while (slot > 0u && topScore[slot - 1u] < value) {
      topScore[slot] = topScore[slot - 1u];
      topWord[slot] = topWord[slot - 1u];
      slot--;
    }
    topScore[slot] = value;
    topWord[slot] = word;
  }
  for (var i = 0u; i < TOP; i++) {
    best[unit * TOP + i] = Best(topWord[i], topScore[i]);
  }
}
`
}
