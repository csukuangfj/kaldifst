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
  rule_ = std::unique_ptr<fst::StdConstFst>(
      CastOrConvertToConstFst(fst::ReadFstKaldiGeneric(rule)));
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
  // Step 1: Convert the input text into an FST
  fst::StdVectorFst text = StringToFst(s);

  // DEBUG: check if rule FST has arc matching first input byte
  if (!s.empty()) {
    uint8_t first_byte = static_cast<uint8_t>(s[0]);
    auto rs = rule_->Start();
    bool found = false;
    int arc_count = 0;
    fst::ArcIterator<fst::StdConstFst> aiter(*rule_, rs);
    for (; !aiter.Done(); aiter.Next()) {
      ++arc_count;
      if (aiter.Value().ilabel == first_byte) {
        found = true;
      }
    }
    fprintf(stderr, "[tn] input='%s' first_byte=%d rule_start=%d arcs_at_start=%d match=%d\n",
            s.c_str(), first_byte, rs, arc_count, found);

    // DEBUG: dump text FST arcs
    fprintf(stderr, "[tn] text FST: start=%d\n", text.Start());
    {
      auto st = text.Start();
      int step = 0;
      while (text.Final(st) == fst::StdArc::Weight::Zero()) {
        fst::ArcIterator<fst::Fst<fst::StdArc>> aiter2(text, st);
        if (aiter2.Done()) break;
        const auto &arc = aiter2.Value();
        fprintf(stderr, "[tn]   text step %d: state=%d ilabel=%d olabel=%d next=%d\n",
                step, st, arc.ilabel, arc.olabel, arc.nextstate);
        st = arc.nextstate;
        ++step;
        aiter2.Next();
        if (!aiter2.Done()) {
          fprintf(stderr, "[tn]   text: NOT LINEAR!\n");
          break;
        }
      }
      fprintf(stderr, "[tn]   text final state=%d is_final=%d\n", st,
              text.Final(st) != fst::StdArc::Weight::Zero());
    }
  }

  // Step 2: Compose the input text with the rule FST
  fst::StdVectorFst composed_fst;
  fst::Compose(text, *rule_, &composed_fst);

  // DEBUG: check composed FST
  fprintf(stderr, "[tn] composed: num_states=%d start=%d\n",
          composed_fst.NumStates(), composed_fst.Start());

  // Check if there's a path from start to a final state
  {
    auto start = composed_fst.Start();
    bool has_any_arc = false;
    int total_arcs = 0;
    for (fst::StateIterator<fst::StdVectorFst> siter(composed_fst); !siter.Done(); siter.Next()) {
      int ns = 0;
      for (fst::ArcIterator<fst::StdVectorFst> aiter(composed_fst, siter.Value()); !aiter.Done(); aiter.Next()) {
        ++ns;
        has_any_arc = true;
      }
      if (siter.Value() == start) {
        fprintf(stderr, "[tn] composed start state %d has %d arcs\n", start, ns);
      }
      total_arcs += ns;
    }
    fprintf(stderr, "[tn] composed total_arcs=%d has_any_arc=%d\n", total_arcs, has_any_arc);

    // Check if any state is final
    int final_count = 0;
    for (fst::StateIterator<fst::StdVectorFst> siter(composed_fst); !siter.Done(); siter.Next()) {
      if (composed_fst.Final(siter.Value()) != fst::StdArc::Weight::Zero()) {
        ++final_count;
      }
    }
    fprintf(stderr, "[tn] composed final_states=%d\n", final_count);
  }

  // Step 3: Get the best path from the composed FST
  fst::StdVectorFst one_best;
  fst::ShortestPath(composed_fst, &one_best, 1);

  fprintf(stderr, "[tn] one_best: num_states=%d\n", one_best.NumStates());

  return FstToString(one_best, remove_output_zero);
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
