# Final system: answer quality

Configuration: BM25 with Greek stemming (k1 = 0.9, b = 0.4), column-aware text, glossary
expansion, chunk 600, k = 5; LLM `groq:openai/gpt-oss-120b`. Produced with

    athex-rag ingest data/raw --embedder none --max-chars 600 --overlap 100 --out indexes/final-bm25
    athex-rag eval eval/questions.jsonl --index indexes/final-bm25 --modes bm25 --llm groq:openai/gpt-oss-120b

and scored with the rounding-consistent number matching (a figure counts only if it is the
true value rounded to the digits shown).

| config | hit@k | answer@k | mrr | correct | correct_numeric | correct_text | hit@k_table_q | correct_table_q | refused_unanswerable | refusal_rate_answerable | citation_precision | citation_support | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25+groq_openai_gpt-oss-120b | 84.4% | 88.3% | 0.600 | 86.0% | 81.0% | 92.9% | 60.7% | 64.3% | 100.0% | 9.1% | 63.3% | 78.8% | 11.34 |

## Error analysis: bm25+groq_openai_gpt-oss-120b

| outcome | count | share | meaning |
|---|---|---|---|
| ok | 61 | 66% | Correct, and cites a source that states the answer |
| ok_bad_citation | 3 | 3% | Correct value, but no cited source states it |
| retrieval_miss | 7 | 8% | No retrieved source states the answer |
| false_refusal | 2 | 2% | The answer was in the sources, but the model said it could not find it |
| wrong_answer | 4 | 4% | The answer was in the sources, but the model got it wrong |
| refused_correctly | 16 | 17% | Unanswerable question, correctly refused |

**retrieval_miss** examples:

- `ote25-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2025;  
  gold: 3.464,3 εκατ. € (p. [190, 212]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p54:t7, OTE_2025:p54:t1, OTE_2025:p117:t1
- `ote25-04` Πόσες ήταν οι ζημιές από διακοπείσες δραστηριότητες του Ομίλου ΟΤΕ το 2025;  
  gold: 224,9 εκατ. € (p. [212]) · got: Οι ζημιές από διακοπείσες δραστηριότητες του Ομίλου ΟΤΕ το 2025 ανέρχονταν σε «Ευρώ 134,7» [3] · retrieved: OTE_2025:p252:t8, OTE_2025:p28:t4, OTE_2025:p260:t7
- `ppc25-04` Ποια ήταν τα καθαρά κέρδη μετά φόρων του Ομίλου ΔΕΗ για το 2025;  
  gold: 359.453 χιλ. € (p. [441]) · got: Τα καθαρά κέρδη μετά φόρων του Ομίλου ΔΕΗ για το 2025 ανέρχονται σε **€0,45 δισ.** [4] · retrieved: PPC_2025:p577:t3, PPC_2025:p476:t0, PPC_2025:p474:t3
- `moh25-02` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Motor Oil το 2025;  
  gold: 11.482.478 χιλ. € (p. [4]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: MOH_2025:p163:t3, MOH_2025:p177:t0, MOH_2025:p185:t3
- `moh25-07` Ποιο ήταν το EBITDA της μητρικής εταιρείας Motor Oil (σε επίπεδο Εταιρείας) το 2025;  
  gold: 745.876 χιλ. € (p. [4]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: MOH_2025:p125:t5, MOH_2025:p172:t1, MOH_2025:p177:t0

**wrong_answer** examples:

- `jumbo24-04` Σε πόσο διαμορφώθηκαν τα αποτελέσματα προ τόκων, φόρων, επενδυτικών αποτελεσμάτων και αποσβέσεων (EBITDA) του Ομίλου Jumbo το 2024;  
  gold: 422,77 εκατ. € (p. [18]) · got: Τα EBITDA του Ομίλου Jumbo το 2024 διαμορφώθηκαν σε **424,55 εκατ. ευρώ**【3】. · retrieved: JUMBO_2024:p18:t3, JUMBO_2024:p25:t2, JUMBO_2024:p25:tab0
- `ote24-03` Ποια ήταν τα κέρδη χρήσης του Ομίλου ΟΤΕ για το 2024;  
  gold: 478,8 εκατ. € (p. [224, 225]) · got: Τα κέρδη χρήσης του Ομίλου ΟΤΕ για το 2024 ανήλθαν σε 492,8 [3] · retrieved: OTE_2024:p224:t6, OTE_2024:p266:t2, OTE_2024:p263:t4
- `ppc24-05` Σε πόσο ανήλθαν οι συνολικές επενδύσεις της ΔΕΗ το 2024;  
  gold: 3 δισ. € (p. [12]) · got: Οι συνολικές επενδύσεις του Ομίλου (ΔΕΗ) για το 2024 ανήλθαν σε **€ 1.875 εκατ.** [1] · retrieved: PPC_2024:p14:t1, PPC_2024:p12:t1, PPC_2024:p14:t4
- `ete24-04` Ποιος ήταν ο Συνολικός Δείκτης Κεφαλαιακής Επάρκειας της Εθνικής Τράπεζας στις 31.12.2024;  
  gold: 21,2% (p. [44]) · got: Ο Συνολικός Δείκτης Κεφαλαιακής Επάρκειας της Εθνικής Τράπεζας στις 31 Δεκεμβρίου 2024 ήταν **21,1 %**【1†L6-L7】【2†L9-L12】 · retrieved: ETE_2024:p536:t4, ETE_2024:p536:tab1, ETE_2024:p44:tab0

**false_refusal** examples:

- `jumbo25-02` Ποια ήταν τα κέρδη μετά από φόρους του Ομίλου Jumbo το 2025;  
  gold: 320.309.417 € (p. [179, 215]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2025:p225:t0, JUMBO_2025:p20:t0, JUMBO_2025:p215:t6
- `ote25-06` Ποια θυγατρική πούλησε ο ΟΤΕ την 1 Οκτωβρίου 2025;  
  gold: Telekom Romania Mobile (p. [19]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p255:t2, OTE_2025:p219:t2, OTE_2025:p218:t7

**ok_bad_citation** examples:

- `ote24-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024, όπως δημοσιεύθηκε στην ετήσια έκθεση 2024;  
  gold: 3.590,8 εκατ. € (p. [200, 224]) · got: Ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024 ανήλθε σε Ευρώ 3,6 δισ. [1] · retrieved: OTE_2024:p62:t9, OTE_2024:p64:t8, OTE_2024:p65:t2
- `ppc25-07` Σε πόσο ανήλθε το προσαρμοσμένο EBITDA της ΔΕΗ το 2025;  
  gold: 2 δισ. € (p. [13]) · got: Το προσαρμοσμένο EBITDA της ΔΕΗ για το 2025 ανήλθε στα €2 δισ. [1][3] · retrieved: PPC_2025:p12:t0, PPC_2025:p13:t3, PPC_2025:p11:t0
- `ete25-03` Ποιος ήταν ο δείκτης MREL της Εθνικής Τράπεζας στο τέλος του 2025;  
  gold: 29,2% (p. [50]) · got: Ο δείκτης MREL της Εθνικής Τράπεζας στις 31 Δεκεμβρίου 2025 ανέρχεται στο 29,2 % επί του TREA【1】【2】. · retrieved: ETE_2025:p54:t7, ETE_2025:p572:t10, ETE_2025:p42:t7
