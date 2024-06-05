#!/usr/bin/env python
# Copyright Johns Hopkins (Amir Hussein)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script measures phonetic similarity distance
"""

import sys
import unicodedata
import numpy as np
import pandas as pd
import tqdm
import os.path

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from statistics import pstdev
import epitran
from abydos.distance import Levenshtein
from abydos.phones import cmp_features, ipa_to_features
import re
import nltk
import string
from nltk.corpus import wordnet
import enchant
from camel_tools.utils.charsets import AR_CHARSET


nltk.download('wordnet')


vowels = ['i',
          'ɪ',
          'ɩ',
          'u',
          'ʊ',
          'ɷ',
          'ᴜ',
          'ɯ̽',
          'ʊ̜',
          'e',
          'ɛ',
          'o',
          'ɤ',
          'ɔ',
          'ʌ',
          'a',
          'a̟',
          'æ̞',
          'ɶ',
          'æ',
          'y',
          'ʏ',
          'ø',
          'œ',
          'ə',
          'ɵ̞',
          'ə̹',
          'ɞ̝',
          'ɘ̞',
          'ɯ',
          'ɒ',
          'ɑ',
          'ɨ',
          'ï',
          'ʉ',
          'ü',
          'ɘ',
          'ë',
          'ɵ',
          'ö',
          'ɜ',
          'ɛ̈',
          'ɞ',
          'ɔ̈',
          'ä',
          'a̠',
          'ɑ̈',
          'ɐ̞',
          'ɐ',
          'ɜ̞',
          'ɞ̞',
          'ɪ̟',
          'ʏ̟',
          'ʏ̫',
          'ʏʷ',
          'ɪʷ',
          'y̫',
          'yʷ',
          'iʷ',
          'u͍',
          'ɯᵝ',
          'ɯ͡β̞',
          'ʉ͍',
          'ɨᵝ',
          'ɨ͡β̞',
          'ɪ̈',
          'ɨ̞',
          'ɘ̝',
          'ʊ̈',
          'ʉ̞',
          'ø̫',
          'øʷ',
          'eʷ',
          'e̞',
          'ɛ̝',
          'ø̞',
          'œ̝',
          'o̞',
          'ɔ̝',
          'ɤ̞',
          'ʌ̝']

regex_pattern = re.compile(r'['+"".join(vowels)+']')


def remove_punctuations(text):
    """ This function  removes all punctuations except the verbatim """

    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    # remove all non verbatim punctuations
    all_punctuations = set(arabic_punctuations + english_punctuations)

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, '')
    return text


def clean_text(text):
    text = re.sub('[أى]', 'ا', text)
    text = re.sub('ة', 'ت', text)
    text = re.sub('[إئ]', 'e', text)
    text = remove_diacritics(text)
    text = remove_punctuations(text)

    return text


def normalize_phone(text, remove_vow=False):
    regex_pattern = re.compile(r'['+"".join(vowels)+']')
    text = re.sub('[ːˤːٔ]', '', text)
    if remove_vow:
        text = re.sub(regex_pattern, '', text)
    return text


_FEATURE_MASK = {
    'syllabic': 3458764513820540928,
    'consonantal': 864691128455135232,
    'sonorant': 216172782113783808,
    'approximant': 54043195528445952,
    'labial': 13510798882111488,
    'round': 3377699720527872,
    'protruded': 844424930131968,
    'compressed': 211106232532992,
    'labiodental': 52776558133248,
    'coronal': 13194139533312,
    'anterior': 3298534883328,
    'distributed': 824633720832,
    'dorsal': 206158430208,
    'high': 51539607552,
    'low': 12884901888,
    'front': 3221225472,
    'back': 805306368,
    'tense': 201326592,
    'pharyngeal': 50331648,
    'atr': 12582912,
    'rtr': 3145728,
    'voice': 786432,
    'spread_glottis': 196608,
    'constricted_glottis': 49152,
    'glottalic_suction': 12288,
    'velaric_suction': 3072,
    'continuant': 768,
    'nasal': 192,
    'strident': 48,
    'lateral': 12,
    'delayed_release': 3,
}


class PhoneticEditDistance(Levenshtein):
    """Phonetic edit distance.

    This is a variation on Levenshtein edit distance, intended for strings in
    IPA, that compares individual phones based on their featural similarity.

    """

    def __init__(
        self,
        mode: str = 'lev',
        cost: Tuple[float, float, float, float] = (1, 1, 1, 0.33333),
        normalizer: Callable[[List[float]], float] = max,
        weights: Optional[Union[Iterable[float], Dict[str, float]]] = None,
        **kwargs: Any
    ):
        """Initialize PhoneticEditDistance instance.

        Parameters
        ----------
        mode : str
            Specifies a mode for computing the edit distance:

                - ``lev`` (default) computes the ordinary Levenshtein distance,
                  in which edits may include inserts, deletes, and
                  substitutions
                - ``osa`` computes the Optimal String Alignment distance, in
                  which edits may include inserts, deletes, substitutions, and
                  transpositions but substrings may only be edited once

        cost : tuple
            A 4-tuple representing the cost of the four possible edits:
            inserts, deletes, substitutions, and transpositions, respectively
            (by default: (1, 1, 1, 0.33333)). Note that transpositions cost a
            relatively low 0.33333. If this were 1.0, no phones would ever be
            transposed under the normal weighting, since even quite dissimilar
            phones such as [a] and [p] still agree in nearly 63% of their
            features.
        normalizer : function
            A function that takes an list and computes a normalization term
            by which the edit distance is divided (max by default). Another
            good option is the sum function.
        weights : None or list or tuple or dict
            If None, all features are of equal significance and a simple
            normalized hamming distance of the features is calculated. If a
            list or tuple of numeric values is supplied, the values are
            inferred as the weights for each feature, in order of the features
            listed in abydos.phones._phones._FEATURE_MASK. If a dict is
            supplied, its key values should match keys in
            abydos.phones._phones._FEATURE_MASK to which each weight (value)
            should be assigned. Missing values in all cases are assigned a
            weight of 0 and will be omitted from the comparison.
        **kwargs
            Arbitrary keyword arguments


        """
        super(PhoneticEditDistance, self).__init__(**kwargs)
        self._mode = mode
        self._cost = cost
        self._normalizer = normalizer

        if isinstance(weights, dict):
            weights = [
                weights[feature] if feature in weights else 0
                for feature in sorted(
                    _FEATURE_MASK, key=_FEATURE_MASK.get, reverse=True
                )
            ]
        elif isinstance(weights, (list, tuple)):
            weights = list(weights) + [0] * (len(_FEATURE_MASK) - len(weights))
        self._weights = weights

    def _alignment_matrix_sim(
        self, src: str, tar: str, backtrace: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the phonetic edit distance alignment matrix.

        Parameters
        ----------
        src : str
            Source string for comparison
        tar : str
            Target string for comparison
        backtrace : bool
            Return the backtrace matrix as well

        Returns
        -------
        numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray)
            The alignment matrix and (optionally) the backtrace matrix



        """

        ins_cost, del_cost, sub_cost, trans_cost = self._cost

        src_len = len(src)
        tar_len = len(tar)

        src_list = ipa_to_features(src)
        tar_list = ipa_to_features(tar)

        d_mat = np.zeros((src_len + 1, tar_len + 1), dtype=np.float_)
        if backtrace:
            trace_mat = np.zeros((src_len + 1, tar_len + 1), dtype=np.int8)
        for i in range(1, src_len + 1):
            d_mat[i, 0] = i * del_cost
            if backtrace:
                trace_mat[i, 0] = 0
        for j in range(1, tar_len + 1):
            d_mat[0, j] = j * ins_cost
            if backtrace:
                trace_mat[0, j] = 1

        for i in range(src_len):
            for j in range(tar_len):
                traces = ((i + 1, j), (i, j + 1), (i, j))
                opts = (
                    d_mat[traces[0]] + ins_cost,  # ins
                    d_mat[traces[1]] + del_cost,  # del
                    d_mat[traces[2]]
                    + (
                        round(sub_cost
                              * (
                                  1.0
                                  - cmp_features(
                                      src_list[i],
                                      tar_list[j],
                                      cast(Sequence[float], self._weights),
                                  )
                              ), 3)
                        if src_list[i] != tar_list[j]
                        else 0
                    ),  # sub/==
                )
                d_mat[i + 1, j + 1] = min(opts)
                if backtrace:
                    trace_mat[i + 1, j + 1] = int(np.argmin(opts))

        if backtrace:
            return d_mat, trace_mat
        return d_mat

    def dist_abs_(self, src: str, tar: str) -> float:
        """Return the phonetic edit distance between two strings.

        Parameters
        ----------
        src : str
            Source string for comparison
        tar : str
            Target string for comparison

        Returns
        -------
        int (may return a float if cost has float values)
            The phonetic edit distance between src & tar
        """

        ins_cost, del_cost, sub_cost, trans_cost = self._cost
        # remove spaces
        src = re.sub('[\s]', '', src)
        tar = re.sub('[\s]', '', tar)
        src_len = len(src)
        tar_len = len(tar)

        if src == tar:
            return 0, 0
        if not src:
            return ins_cost * tar_len, ins_cost * tar_len
        if not tar:
            return del_cost * src_len, del_cost * src_len

        d_mat = cast(
            np.ndarray, self._alignment_matrix_sim(src, tar, backtrace=False)
        )
        d_mat_per = cast(
            np.ndarray, self.alignment_matrix(src, tar, backtrace=False)
        )

        return cast(float, d_mat[src_len, tar_len]), cast(float, d_mat_per[src_len, tar_len])

    def alignment_matrix(
        self, src: str, tar: str, backtrace: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the phonetic edit distance alignment matrix.

        Parameters
        ----------
        src : str
            Source string for comparison
        tar : str
            Target string for comparison
        backtrace : bool
            Return the backtrace matrix as well

        Returns
        -------
        numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray)
            The alignment matrix and (optionally) the backtrace matrix

        """
        ins_cost, del_cost, sub_cost, trans_cost = self._cost

        src_len = len(src)
        tar_len = len(tar)

        d_mat = np.zeros((src_len + 1, tar_len + 1), dtype=np.float_)
        if backtrace:
            trace_mat = np.zeros((src_len + 1, tar_len + 1), dtype=np.int8)
        for i in range(1, src_len + 1):
            d_mat[i, 0] = i * del_cost
            if backtrace:
                trace_mat[i, 0] = 0
        for j in range(1, tar_len + 1):
            d_mat[0, j] = j * ins_cost
            if backtrace:
                trace_mat[0, j] = 1

        for i in range(src_len):
            for j in range(tar_len):
                traces = ((i + 1, j), (i, j + 1), (i, j))
                opts = (
                    d_mat[traces[0]] + 1,  # ins
                    d_mat[traces[1]] + 1,  # del
                    d_mat[traces[2]] + (1
                                        if src[i] != tar[j]
                                        else 0)
                )
                d_mat[i + 1, j + 1] = min(opts)
                if backtrace:
                    trace_mat[i + 1, j + 1] = int(np.argmin(opts))
        if backtrace:
            return d_mat, trace_mat
        return d_mat


def remove_diacritics(text):
    # https://unicode-table.com/en/blocks/arabic/
    return re.sub(r'[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+', '', text)


def remove_en_characters(text):
    return re.sub(r'[^\u0600-\u06FF\s]+', '', text)


def remove_ar_characters(text):
    return re.sub(r'[\u0600-\u06FF\s]+', '', text)


def read_tsv(data_file):
    text_data = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        line = line.strip()
        # text = line.split('\t')
        text_data.append(line)
    return text_data


def isEnglish(word):
    # This function takes a word and checks if it's an English word
    # enchant_US_dict = enchant.Dict("en_US")
    # return (enchant_US_dict.check(word) or bool(wordnet.synsets(word))) and (word not in string.punctuation)
    english_reg = re.compile('^[a-zA-Z]+$')
    return (english_reg.match(word) is not None)


def isArabic(word):
    ar_str = u''.join(AR_CHARSET)
    arabic_re = re.compile(r'^[' + re.escape(ar_str) + r']+$')
    return (arabic_re.match(word) is not None)


def phonetic_sim_dist(phonetic, text_hyp, text_ref, epi_en, epi_ar, debug=False):
    """
    This function calculates the  similarity between the strings
    of phones

    args:

        phonetic (PhoneticEditDistance object)
        text_hyp (str):  arabic phones
        text_ref (str): english phones
        norm_diac (bool):  normalize the phonetic diacritics
        remove_vow (bool): remove vowels
    return:
        phone smilarity distance (float)
    """
    text_hyp = unicodedata.normalize('NFC', clean_text(text_hyp.lower()))
    text_ref = unicodedata.normalize('NFC', clean_text(text_ref.lower()))

    hyp_ = []
    ref_ = []
    hyp_norm = []
    ref_norm = []
    for token in text_hyp.split():

        if isArabic(token):
            hyp_norm.append(normalize_phone(
                epi_ar.transliterate(token), remove_vow=True))
            hyp_.append(normalize_phone(epi_ar.transliterate(token)))
        elif isEnglish(token):
            hyp_norm.append(normalize_phone(
                epi_en.transliterate(token), remove_vow=True))
            hyp_.append(normalize_phone(epi_en.transliterate(token)))
        else:
            new_token = []
            new_token_norm = []
            for subtok in re.findall(r"([a-z']+|[^a-z']+)", token):
                if remove_en_characters(subtok) != '':
                    new_token.append(normalize_phone(epi_ar.transliterate(
                        subtok)))
                    new_token_norm.append(normalize_phone(
                        epi_ar.transliterate(subtok), remove_vow=True))
                elif remove_ar_characters(subtok) != '':
                    new_token.append(normalize_phone(epi_en.transliterate(
                        subtok)))
                    new_token_norm.append(normalize_phone(
                        epi_en.transliterate(subtok), remove_vow=True))
            hyp_.append("".join(new_token))
            hyp_norm.append("".join(new_token_norm))

    for token in text_ref.split():

        if isArabic(token):
            ref_norm.append(normalize_phone(
                epi_ar.transliterate(token), remove_vow=True))
            ref_.append(normalize_phone(epi_ar.transliterate(token)))

        elif isEnglish(token):
            ref_norm.append(normalize_phone(
                epi_en.transliterate(token), remove_vow=True))
            ref_.append(normalize_phone(epi_en.transliterate(token)))
        else:
            new_token = []
            new_token_norm = []
            for subtok in re.findall(r"([a-z']+|[^a-z']+)", token):
                if remove_en_characters(subtok) != '':
                    new_token.append(normalize_phone(epi_ar.transliterate(
                        subtok)))
                    new_token_norm.append(normalize_phone(
                        epi_ar.transliterate(subtok), remove_vow=True))
                elif remove_ar_characters(subtok) != '':
                    new_token.append(normalize_phone(epi_en.transliterate(
                        subtok)))
                    new_token_norm.append(normalize_phone(
                        epi_en.transliterate(subtok), remove_vow=True))
            ref_.append("".join(new_token))
            ref_norm.append("".join(new_token_norm))

    tar_len = len("".join(ref_))
    tar_len_norm = len("".join(ref_norm))
    hyp = " ".join(hyp_)
    ref = " ".join(ref_)
    hyp_norm = " ".join(hyp_norm)
    ref_norm = " ".join(ref_norm)

    psd, per = phonetic.dist_abs_(hyp, ref)
    psd_norm, _ = phonetic.dist_abs_(hyp_norm, ref_norm)

    # if debug == True:
    #     print("Ar: ", hyp)
    #     print("En: ", ref)
    #     print("Psd: ", round(phonetic.dist_abs(hyp, ref), 5)/tar_len)
    #     print("Psd_norm: ", round(phonetic.dist_abs(
    #         hyp_norm, ref_norm), 5)/tar_len_norm)

    return [psd, psd_norm, per], [tar_len, tar_len_norm, tar_len], hyp, ref


def main():

    hyp_file = sys.argv[1]  # hyp transcription file with format:  <id> <text>
    ref_file = sys.argv[2]  # ref transcription file with format:  <id> <text>
    hyp_data = read_tsv(hyp_file)
    ref_data = read_tsv(ref_file)

    hyp_dict = {}
    ref_dict = {}
    total_len = 0
    total_norm_len = 0
    PER_tot = 0
    PSD_tot = 0
    PSD_norm_tot = 0
    epi_en = epitran.Epitran('eng-Latn', ligatures=True)
    epi_ar = epitran.Epitran('ara-Arab', ligatures=True)
    # read hyp
    for i in range(len(hyp_data)):
        #
        hyp_dict[hyp_data[i].split()[0]] = " ".join(hyp_data[i].split()[1:])

    for i in range(len(ref_data)):

        ref_dict[ref_data[i].split()[0]] = " ".join(ref_data[i].split()[1:])

    phonetic = PhoneticEditDistance(cost=(1, 1, 4, 0.33333))
    hyp_name = os.path.basename(hyp_file).split('.')[0]
    ref_name = os.path.basename(ref_file).split('.')[0]
    with open(f"results_{hyp_name}_{ref_name}.txt", "w") as f, open(f'tab_sep_res_{hyp_name}_{ref_name}.txt', 'w') as b:
        for key in tqdm.tqdm(hyp_dict):
            # print(hyp_dict[key], ref_dict[key])

            metrics, lengths, hyp_phone, ref_phone = phonetic_sim_dist(
                phonetic, hyp_dict[key], ref_dict[key], epi_en, epi_ar)

            f.write("ID: " + key+"\n")
            f.write("REF: " + ref_dict[key]+"\n")
            f.write("HYP: " + hyp_dict[key]+"\n")
            f.write("REF phone: " + ref_phone + "\n")
            f.write("HYP phone: " + hyp_phone + "\n")
            f.write(
                "PER: " + str(round(metrics[2]/lengths[2], 5) if lengths[2] != 0 else metrics[2])+" ")
            f.write(
                "PSD: " + str(round(metrics[0]/lengths[0], 5) if lengths[0] != 0 else metrics[0])+" ")
            f.write(
                "PSD_norm: " + str(round(metrics[1]/lengths[1], 5) if lengths[1] != 0 else metrics[1])+"\n\n")

            b.write(str(round(metrics[2]/lengths[2], 5)
                    if lengths[2] != 0 else metrics[2])+"\t")
            b.write(str(round(metrics[0]/lengths[0], 5)
                    if lengths[0] != 0 else metrics[0])+"\t")
            b.write(str(round(metrics[1]/lengths[1], 5)
                    if lengths[1] != 0 else metrics[1])+"\n")

            PER_tot += metrics[2]
            PSD_tot += metrics[0]
            PSD_norm_tot += metrics[1]
            total_len += lengths[0]
            total_norm_len += lengths[1]

        f.write("\n\n")
        f.write("Mean PER_tot:     " + str(round(PER_tot/total_len, 5))+"\n ")
        f.write("Mean PSD_tot:      " + str(round(PSD_tot/total_len, 5))+"\n ")
        f.write("Mean PSD_norm_tot: " +
                str(round(PSD_norm_tot/total_norm_len, 5))+"\n")


if __name__ == "__main__":
    main()
