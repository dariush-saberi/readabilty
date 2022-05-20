#!/usr/bin/env python
# coding: utf-8
import textstat
from textstat import textstat as tx

def get_stat(test_data):
    a = textstat.flesch_reading_ease(test_data)
    b = textstat.flesch_kincaid_grade(test_data)
    c = textstat.smog_index(test_data)
    d = textstat.coleman_liau_index(test_data)
    e = textstat.automated_readability_index(test_data)
    f = textstat.dale_chall_readability_score(test_data)
    g = textstat.difficult_words(test_data)
    h = textstat.linsear_write_formula(test_data)
    i = textstat.gunning_fog(test_data)
    j = textstat.text_standard(test_data)
    k = textstat.fernandez_huerta(test_data)
    l = textstat.szigriszt_pazos(test_data)
    m = textstat.gutierrez_polini(test_data)
    n = textstat.crawford(test_data)
    o = textstat.gulpease_index(test_data)
    p = textstat.osman(test_data)
    x = tx.difficult_words_list(test_data)
    
    result = {"Flesch Reading Ease":a,
              "Flesch Kincaid Grade":b,
              "SMOG Index":c,
              "Coleman Liau Index":d,
              "Automated Readability Index":e,
              "Dale Chall Readability Score":f,
              "Difficult Words":g,
              "Linsear Write Formula":h,
              "Gunning Fog":i,
              "Text Standard":j,
              "Fernandez Huerta":k,
              "Szigriszt Pazos":l,
              "Gutierrez Polini":m,
              "Crawford":n,
              "Gulpease Index":o,
              "Osman":p,
              "Words List":x}
    return result
