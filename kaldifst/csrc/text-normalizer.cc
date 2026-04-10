// kaldifst/csrc/text-normalizer.cc
//
// Copyright (c)  2023 Xiaomi Corporation

#include "kaldifst/csrc/text-normalizer.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/table-matcher.h"

namespace fst {

// This variable is copied from
// https://github.com/pzelasko/Pynini/blob/master/src/stringcompile.h#L81
constexpr uint64_t kCompiledStringProps =
    kAcceptor | kIDeterministic | kODeterministic | kILabelSorted |
    kOLabelSorted | kUnweighted | kAcyclic | kInitialAcyclic | kTopSorted |
    kAccessible | kCoAccessible | kString | kUnweightedCycles;
}  // namespace fst

namespace kaldifst {

// We don't use StringCompiler<StdArc> here since it treats bytes as
// signed integers.
static fst::StdVectorFst StringToFst(const std::string &text) {
  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;

  fst::StdVectorFst ans;
  ans.ReserveStates(text.size());

  auto s = ans.AddState();
  ans.SetStart(s);
  // CAUTION(fangjun): We need to use uint8_t here.
  for (const uint8_t label : text) {
    const auto nextstate = ans.AddState();
    ans.AddArc(s, Arc(label, label, Weight::One(), nextstate));
    s = nextstate;
  }

  ans.SetFinal(s, Weight::One());
  ans.SetProperties(fst::kCompiledStringProps, fst::kCompiledStringProps);

  return ans;
}

static fst::StdVectorFst StringToFst(
    const std::vector<std::string> &words,
    const std::vector<std::string> &pronunciations) {
  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;

  fst::StdVectorFst ans;

  auto s = ans.AddState();
  ans.SetStart(s);

  int32_t n = words.size();
  for (int32_t i = 0; i != n; ++i) {
    const auto &w = words[i];
    const auto &p = pronunciations[i];
    const uint8_t *w_ptr = reinterpret_cast<const uint8_t *>(w.data());
    const uint8_t *p_ptr = reinterpret_cast<const uint8_t *>(p.data());

    int32_t max_size = std::max<int32_t>(w.size(), p.size());

    for (int32_t k = 0; k < max_size; ++k) {
      uint8_t i_label = k < w.size() ? w_ptr[k] : 0;
      uint8_t o_label = k < p.size() ? p_ptr[k] : 0;

      const auto nextstate = ans.AddState();
      ans.AddArc(s, Arc(i_label, o_label, Weight::One(), nextstate));
      s = nextstate;
    }
  }

  ans.SetFinal(s, Weight::One());
  ans.SetProperties(fst::kCompiledStringProps, fst::kCompiledStringProps);

  return ans;
}

static std::string FstToString(const fst::StdVectorFst &fst,
                               bool remove_output_zero) {
  std::string ans;

  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;
  auto s = fst.Start();
  if (s == fst::kNoStateId) {
    // this is an empty FST
    return "";
  }
  while (fst.Final(s) == Weight::Zero()) {
    fst::ArcIterator<fst::Fst<Arc>> aiter(fst, s);
    if (aiter.Done()) {
      // not reached final.
      return "";
    }

    const auto &arc = aiter.Value();
    if (arc.olabel != 0 || !remove_output_zero) {
      ans.push_back(arc.olabel);
    }

    s = arc.nextstate;
    if (s == fst::kNoStateId) {
      // Transition to invalid state";
      return "";
    }

    aiter.Next();
    if (!aiter.Done()) {
      // not a linear FST
      return "";
    }
  }
  return ans;
}

static std::string FstToString2(const fst::StdVectorFst &fst) {
  std::string ans;

  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;
  auto s = fst.Start();
  if (s == fst::kNoStateId) {
    // this is an empty FST
    return "";
  }

  while (fst.Final(s) == Weight::Zero()) {
    fst::ArcIterator<fst::Fst<Arc>> aiter(fst, s);
    if (aiter.Done()) {
      // not reached final.
      return "";
    }

    const auto &arc = aiter.Value();

    if (arc.olabel != 0 && arc.olabel < 128) {
      if (arc.ilabel != 0) {
        ans.push_back(arc.ilabel);
      }
    } else if (arc.olabel >= 128) {
      ans.push_back(arc.olabel);
    }

    s = arc.nextstate;
    if (s == fst::kNoStateId) {
      // Transition to invalid state";
      return "";
    }

    aiter.Next();
    if (!aiter.Done()) {
      // not a linear FST
      return "";
    }
  }

  return ans;
}

TextNormalizer::TextNormalizer(const std::string &rule) {
  auto *raw = fst::ReadFstKaldiGeneric(rule);
  fprintf(stderr, "[kaldifst] TextNormalizer ctor: rule='%s' raw=%p\n", rule.c_str(), (void*)raw);
  rule_ = std::unique_ptr<fst::StdConstFst>(CastOrConvertToConstFst(raw));
  if (rule_) {
    fprintf(stderr, "[kaldifst]   const fst NumStates=%lld Start=%d\n",
            (long long)rule_->NumStates(), rule_->Start());
  } else {
    fprintf(stderr, "[kaldifst]   const fst is NULL!\n");
  }
}

TextNormalizer::TextNormalizer(std::istream &is) {
  fst::StdVectorFst *fst = new fst::StdVectorFst;
  bool binary = true;
  ReadFstKaldi(is, binary, fst);

  // fst is released inside CastOrConvertToConstFst()
  rule_ = std::unique_ptr<fst::StdConstFst>(CastOrConvertToConstFst(fst));
}

TextNormalizer::TextNormalizer(std::unique_ptr<fst::StdConstFst> rule)
    : rule_(std::move(rule)) {}

std::string TextNormalizer::Normalize(const std::string &s,
                                      bool remove_output_zero /*=true*/) const {
  fprintf(stderr, "[kaldifst] Normalize input='%s' (len=%zu) rule_=%p\n",
          s.c_str(), s.size(), (void*)rule_.get());
  if (!rule_) {
    fprintf(stderr, "[kaldifst]   rule_ is null, returning empty\n");
    return "";
  }

  // Print first 5 arcs of the rule FST at the start state
  {
    auto rs = rule_->Start();
    fprintf(stderr, "[kaldifst]   rule start state=%d\n", rs);
    fst::ArcIterator<fst::StdConstFst> aiter(*rule_, rs);
    int count = 0;
    for (; !aiter.Done() && count < 10; aiter.Next(), ++count) {
      const auto &arc = aiter.Value();
      fprintf(stderr, "[kaldifst]     rule arc[%d]: ilabel=%d olabel=%d nextstate=%d\n",
              count, arc.ilabel, arc.olabel, arc.nextstate);
    }
  }

  // Step 1: Convert the input text into an FST
  fst::StdVectorFst text = StringToFst(s);
  fprintf(stderr, "[kaldifst]   text FST: NumStates=%lld Start=%d\n",
          (long long)text.NumStates(), text.Start());
  // Print text FST arcs
  for (int64_t i = 0; i < std::min((long long)text.NumStates(), 5LL); ++i) {
    fst::ArcIterator<fst::StdVectorFst> aiter(text, i);
    for (int count = 0; !aiter.Done() && count < 3; aiter.Next(), ++count) {
      const auto &arc = aiter.Value();
      fprintf(stderr, "[kaldifst]     text arc[%lld]: ilabel=%d olabel=%d nextstate=%d\n",
              (long long)i, arc.ilabel, arc.olabel, arc.nextstate);
    }
  }

  // Step 2: Compose the input text with the rule FST
  fst::StdVectorFst composed_fst;
  fst::Compose(text, *rule_, &composed_fst);
  fprintf(stderr, "[kaldifst]   composed FST: NumStates=%lld Start=%d\n",
          (long long)composed_fst.NumStates(), composed_fst.Start());

  // Step 3: Get the best path from the composed FST
  fst::StdVectorFst one_best;
  fst::ShortestPath(composed_fst, &one_best, 1);
  fprintf(stderr, "[kaldifst]   one_best FST: NumStates=%lld Start=%d\n",
          (long long)one_best.NumStates(), one_best.Start());

  std::string result = FstToString(one_best, remove_output_zero);
  fprintf(stderr, "[kaldifst]   result='%s' (len=%zu)\n", result.c_str(), result.size());
  return result;
}

std::string TextNormalizer::Normalize(
    const std::vector<std::string> &words,
    const std::vector<std::string> &pronunciations) const {
  if (words.size() != pronunciations.size()) {
    return {};
  }

  // Step 1: Convert the input text into an FST
  fst::StdVectorFst text = StringToFst(words, pronunciations);

  // Step 2: Compose the input text with the rule FST
  fst::StdVectorFst composed_fst;
  fst::Compose(text, *rule_, &composed_fst);

  // Step 3: Get the best path from the composed FST
  fst::StdVectorFst one_best;
  fst::ShortestPath(composed_fst, &one_best, 1);

  return FstToString2(one_best);
}

}  // namespace kaldifst
