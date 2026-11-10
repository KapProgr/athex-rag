# Answer quality and retrieval (rescored)

All runs from `athex-rag experiments` (bm25 / dense / hybrid at chunk 600, 1200 and 2000; LLM runs
with groq:openai/gpt-oss-120b and groq:qwen/qwen3.8-27b on the best retrieval configuration),
re-scored with `athex-rag rescore` after two measurement fixes: citations written as 【1】 /
【1†L1-L3】 are parsed, and a retrieved or cited chunk counts as supporting the answer if it
states it, even on a page other than the gold page. No answers were regenerated.

Re-scored again with the stricter, rounding-consistent number matching (see final_answers.md).

| config | hit@k | answer@k | mrr | correct | correct_numeric | correct_text | hit@k_table_q | correct_table_q | refused_unanswerable | refusal_rate_answerable | citation_precision | citation_support | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-llm-bm25-chunk600-groq-openai-gpt-oss-120b | 71.4% | 80.5% | 0.528 | 81.7% | 82.5% | 57.1% | 46.4% | 64.3% | 100.0% | 16.9% | 56.6% | 78.8% | 11.13 |
| 5-llm-bm25-chunk600-groq-qwen-qwen3.8-27b | 71.4% | 80.5% | 0.528 | 83.9% | 85.7% | 57.1% | 46.4% | 71.4% | 100.0% | 15.6% | 55.6% | 77.2% | 12.39 |

## Error analysis: 5-llm-bm25-chunk600-groq-openai-gpt-oss-120b

| outcome | count | share | meaning |
|---|---|---|---|
| ok | 56 | 60% | Correct, and cites a source that states the answer |
| ok_bad_citation | 4 | 4% | Correct value, but no cited source states it |
| retrieval_miss | 12 | 13% | No retrieved source states the answer |
| false_refusal | 2 | 2% | The answer was in the sources, but the model said it could not find it |
| wrong_answer | 3 | 3% | The answer was in the sources, but the model got it wrong |
| refused_correctly | 16 | 17% | Unanswerable question, correctly refused |

**retrieval_miss** examples:

- `jumbo25-06` Ποια ελεγκτική εταιρεία υπογράφει την έκθεση ελέγχου της Jumbo για τη χρήση 2025;  
  gold: BDO (p. [16, 171]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2025:p67:t2, JUMBO_2025:p259:t0, JUMBO_2025:p168:t0
- `jumbo24-06` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Jumbo για τη χρήση 2024;  
  gold: BDO (p. [15, 165]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2024:p6:t0, JUMBO_2024:p17:t3, JUMBO_2024:p17:t0
- `ote25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2025;  
  gold: Deloitte (p. [199, 207]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p201:t0, OTE_2025:p148:t3, OTE_2025:p273:t2
- `ote25-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2025;  
  gold: 3.464,3 εκατ. € (p. [190, 212]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p54:t7, OTE_2025:p54:t1, OTE_2025:p54:t6
- `ote25-04` Πόσες ήταν οι ζημιές από διακοπείσες δραστηριότητες του Ομίλου ΟΤΕ το 2025;  
  gold: 224,9 εκατ. € (p. [212]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p252:t8, OTE_2025:p28:t4, OTE_2025:p55:t5

**wrong_answer** examples:

- `ote25-03` Ποια ήταν τα κέρδη χρήσης του Ομίλου ΟΤΕ για το 2025;  
  gold: 501,1 εκατ. € (p. [212, 213]) · got: Τα κέρδη χρήσης του Ομίλου ΟΤΕ για το 2025 ήταν: - **Κέρδη χρήσης από συνεχιζόμενες δραστηριότητες**: 726,0 [3][4] - **Κέρδη χρήσης από διακοπείσες δραστηριότητες**: (224,9) [3][4] · retrieved: OTE_2025:p164:t5, OTE_2025:p148:t3, OTE_2025:p280:t1
- `ppc24-05` Σε πόσο ανήλθαν οι συνολικές επενδύσεις της ΔΕΗ το 2024;  
  gold: 3 δισ. € (p. [12]) · got: Οι συνολικές επενδύσεις του Ομίλου ΔΕΗ για το 2024 ανήλθαν σε **€ 1,875 εκατ.** [1] · retrieved: PPC_2024:p14:t1, PPC_2024:p12:t1, PPC_2024:p577:t6
- `ete24-05` Ποιος ήταν ο Δείκτης Κάλυψης Ρευστότητας (LCR) της Εθνικής Τράπεζας στο τέλος του 2024;  
  gold: 261% (p. [44]) · got: Ο Δείκτης Κάλυψης Ρευστότητας (LCR) της Εθνικής Τράπεζας στο τέλος του 2024 ήταν **262 %** [2] · retrieved: ETE_2024:p411:t3, ETE_2024:p44:t2, ETE_2024:p101:t1

**false_refusal** examples:

- `ote25-06` Ποια θυγατρική πούλησε ο ΟΤΕ την 1 Οκτωβρίου 2025;  
  gold: Telekom Romania Mobile (p. [19]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p255:t2, OTE_2025:p219:t2, OTE_2025:p20:t7
- `ote24-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2024;  
  gold: Deloitte (p. [205, 209, 211, 219]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2024:p211:t1, OTE_2024:p282:t10, OTE_2024:p159:t1

**ok_bad_citation** examples:

- `jumbo25-05` Πόσους εργαζόμενους είχε η θυγατρική της Jumbo στη Ρουμανία στο τέλος του 2025;  
  gold: 1.750 άτομα (p. [36, 249]) · got: 1.750 εργαζόμενοι στη ρουμανική θυγατρική της Jumbo (αριθμός εργαζομένων 31 Δεκεμβρίου 2025)【3】 · retrieved: JUMBO_2025:p216:t0, JUMBO_2025:p27:t0, JUMBO_2025:p95:t2
- `ote24-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024, όπως δημοσιεύθηκε στην ετήσια έκθεση 2024;  
  gold: 3.590,8 εκατ. € (p. [200, 224]) · got: Ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024 ήταν Ευρώ 3,6 δισ. [2] · retrieved: OTE_2024:p65:t2, OTE_2024:p62:t9, OTE_2024:p64:t8
- `ppc25-07` Σε πόσο ανήλθε το προσαρμοσμένο EBITDA της ΔΕΗ το 2025;  
  gold: 2 δισ. € (p. [13]) · got: Το προσαρμοσμένο EBITDA της ΔΕΗ το 2025 ανήλθε στα €2 δισ. [1][5] · retrieved: PPC_2025:p12:t0, PPC_2025:p13:t3, PPC_2025:p437:t6
- `ete25-03` Ποιος ήταν ο δείκτης MREL της Εθνικής Τράπεζας στο τέλος του 2025;  
  gold: 29,2% (p. [50]) · got: Ο δείκτης MREL της Εθνικής Τράπεζας στις 31 Δεκεμβρίου 2025 ανέρχεται στο 29,2 % επί του TREA [1][2]. · retrieved: ETE_2025:p54:t7, ETE_2025:p572:t10, ETE_2025:p572:t5

## Error analysis: 5-llm-bm25-chunk600-groq-qwen-qwen3.8-27b

| outcome | count | share | meaning |
|---|---|---|---|
| ok | 58 | 62% | Correct, and cites a source that states the answer |
| ok_bad_citation | 4 | 4% | Correct value, but no cited source states it |
| retrieval_miss | 12 | 13% | No retrieved source states the answer |
| false_refusal | 2 | 2% | The answer was in the sources, but the model said it could not find it |
| wrong_answer | 1 | 1% | The answer was in the sources, but the model got it wrong |
| refused_correctly | 16 | 17% | Unanswerable question, correctly refused |

**retrieval_miss** examples:

- `jumbo25-06` Ποια ελεγκτική εταιρεία υπογράφει την έκθεση ελέγχου της Jumbo για τη χρήση 2025;  
  gold: BDO (p. [16, 171]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2025:p67:t2, JUMBO_2025:p259:t0, JUMBO_2025:p168:t0
- `jumbo24-06` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Jumbo για τη χρήση 2024;  
  gold: BDO (p. [15, 165]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2024:p6:t0, JUMBO_2024:p17:t3, JUMBO_2024:p17:t0
- `ote25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2025;  
  gold: Deloitte (p. [199, 207]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p201:t0, OTE_2025:p148:t3, OTE_2025:p273:t2
- `ote25-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2025;  
  gold: 3.464,3 εκατ. € (p. [190, 212]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p54:t7, OTE_2025:p54:t1, OTE_2025:p54:t6
- `ote25-04` Πόσες ήταν οι ζημιές από διακοπείσες δραστηριότητες του Ομίλου ΟΤΕ το 2025;  
  gold: 224,9 εκατ. € (p. [212]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p252:t8, OTE_2025:p28:t4, OTE_2025:p55:t5

**wrong_answer** examples:

- `ppc24-05` Σε πόσο ανήλθαν οι συνολικές επενδύσεις της ΔΕΗ το 2024;  
  gold: 3 δισ. € (p. [12]) · got: Οι συνολικές επενδύσεις του Ομίλου ανήλθαν στο ποσό των € 1.875εκατ. για το έτος 2024 [1]. · retrieved: PPC_2024:p14:t1, PPC_2024:p12:t1, PPC_2024:p577:t6

**false_refusal** examples:

- `ote25-06` Ποια θυγατρική πούλησε ο ΟΤΕ την 1 Οκτωβρίου 2025;  
  gold: Telekom Romania Mobile (p. [19]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p255:t2, OTE_2025:p219:t2, OTE_2025:p20:t7
- `ote24-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2024;  
  gold: Deloitte (p. [205, 209, 211, 219]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2024:p211:t1, OTE_2024:p282:t10, OTE_2024:p159:t1

**ok_bad_citation** examples:

- `jumbo25-05` Πόσους εργαζόμενους είχε η θυγατρική της Jumbo στη Ρουμανία στο τέλος του 2025;  
  gold: 1.750 άτομα (p. [36, 249]) · got: Η θυγατρική της Jumbo στη Ρουμανία είχε 1.750 εργαζόμενους στο τέλος του 2025 [3]. · retrieved: JUMBO_2025:p216:t0, JUMBO_2025:p27:t0, JUMBO_2025:p95:t2
- `ote24-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024, όπως δημοσιεύθηκε στην ετήσια έκθεση 2024;  
  gold: 3.590,8 εκατ. € (p. [200, 224]) · got: Ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ για το οικονομικό έτος 2024 ανήλθε σε Ευρώ 3,6 δισ. [2] · retrieved: OTE_2024:p65:t2, OTE_2024:p62:t9, OTE_2024:p64:t8
- `ppc25-07` Σε πόσο ανήλθε το προσαρμοσμένο EBITDA της ΔΕΗ το 2025;  
  gold: 2 δισ. € (p. [13]) · got: Το προσαρμοσμένο EBITDA της ΔΕΗ το 2025 ανήλθε στα €2 δισ. [1][5] · retrieved: PPC_2025:p12:t0, PPC_2025:p13:t3, PPC_2025:p437:t6
- `ete25-03` Ποιος ήταν ο δείκτης MREL της Εθνικής Τράπεζας στο τέλος του 2025;  
  gold: 29,2% (p. [50]) · got: Στις 31 Δεκεμβρίου 2025, ο δείκτης MREL της Τράπεζας σε ενοποιημένο επίπεδο ανέρχεται στο 29,2% επί του TREA [1][2]. · retrieved: ETE_2025:p54:t7, ETE_2025:p572:t10, ETE_2025:p572:t5
