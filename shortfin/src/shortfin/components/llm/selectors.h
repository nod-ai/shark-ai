#ifndef SHORTFIN_COMPONENTS_LLM_SELECTORS_H
#define SHORTFIN_COMPONENTS_LLM_SELECTORS_H

#include <vector>

#include "shortfin/components/llm/data.h"

namespace shortfin::llm {

SHORTFIN_API void SelectTokens(const std::vector<float> &scores,
                               const DecodeConfig &config,
                               std::vector<int> &selected_tokens,
                               std::vector<float> &selected_scores);

}  // namespace shortfin::llm

#endif
