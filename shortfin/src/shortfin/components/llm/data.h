#ifndef SHORTFIN_COMPONENTS_LLM_DATA_H
#define SHORTFIN_COMPONENTS_LLM_DATA_H

#include "shortfin/support/api.h"

namespace shortfin::llm {

enum class LogitsNormalization {
  NONE,
  SOFTMAX,
  LOG_SOFTMAX,
};

// Configuration for decoding
struct SHORTFIN_API DecodeConfig {
  int eos_token_id = 2;
  int num_beams = 1;
  LogitsNormalization logits_normalization = LogitsNormalization::NONE;
  int max_completion_tokens = 50;
  float temperature = 0.8f;
  bool use_beam_search = false;
  int top_k = -1;       // -1 means no top_k filtering
  float top_p = -1.0f;  // -1 means no top_p filtering
};

}  // namespace shortfin::llm

#endif
