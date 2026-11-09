> Raw output of the first `athex-rag experiments` run. Citation and support metrics here predate two
> measurement fixes; see [answer_quality.md](answer_quality.md) for the corrected numbers.

# Experiment results

93 questions (77 answerable, 16 unanswerable) over 10 reports. k = 5. Best retrieval configuration: **bm25 · chunk600**.

## 1. Chunk size × retrieval mode

| config | hit@k | mrr | hit@k_table_q | latency_s |
|---|---|---|---|---|
| bm25 · chunk600 | 71.4% | 0.528 | 46.4% | 0.09 |
| dense · multilingual-e5-small · chunk600 | 61.0% | 0.482 | 35.7% | 0.04 |
| hybrid · multilingual-e5-small · chunk600 | 67.5% | 0.512 | 39.3% | 0.12 |
| bm25 · chunk1200 | 68.8% | 0.523 | 50.0% | 0.05 |
| dense · multilingual-e5-small · chunk1200 | 58.4% | 0.452 | 39.3% | 0.03 |
| hybrid · multilingual-e5-small · chunk1200 | 67.5% | 0.525 | 46.4% | 0.07 |
| bm25 · chunk2000 | 68.8% | 0.518 | 50.0% | 0.04 |
| dense · multilingual-e5-small · chunk2000 | 58.4% | 0.464 | 28.6% | 0.03 |
| hybrid · multilingual-e5-small · chunk2000 | 68.8% | 0.499 | 42.9% | 0.06 |

## 2. Table extraction

| config | hit@k | mrr | hit@k_table_q | latency_s |
|---|---|---|---|---|
| bm25 · chunk600 | 71.4% | 0.528 | 46.4% | 0.09 |
| bm25 · chunk600 · no-tables | 68.8% | 0.524 | 39.3% | 0.10 |

## 3. Embedding model

| config | hit@k | mrr | hit@k_table_q | latency_s |
|---|---|---|---|---|
| bm25 · chunk600 | 71.4% | 0.528 | 46.4% | 0.09 |

## 5. Answer generation

| config | hit@k | mrr | correct | correct_numeric | correct_text | hit@k_table_q | correct_table_q | refused_unanswerable | refusal_rate_answerable | citation_precision | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 · chunk600 · groq:openai/gpt-oss-120b | 71.4% | 0.528 | 81.7% | 84.1% | 50.0% | 46.4% | 67.9% | 100.0% | 16.9% | 62.1% | 11.13 |
| bm25 · chunk600 · groq:qwen/qwen3.8-27b | 71.4% | 0.528 | 83.9% | 85.7% | 57.1% | 46.4% | 71.4% | 100.0% | 15.6% | 55.6% | 12.39 |

## Error analysis: bm25 · chunk600 · groq:openai/gpt-oss-120b

| outcome | count | share | meaning |
|---|---|---|---|
| ok | 32 | 34% | Correct, and cites the right page |
| ok_bad_citation | 28 | 30% | Correct value, but no citation on the gold page |
| retrieval_miss | 15 | 16% | Gold page not in the retrieved sources |
| false_refusal | 1 | 1% | Gold page retrieved, but the model said it could not find the answer |
| wrong_answer | 1 | 1% | Gold page retrieved, but the answer is wrong (unit, year, misread table) |
| refused_correctly | 16 | 17% | Unanswerable question, correctly refused |

**retrieval_miss** examples:

- `jumbo25-06` Ποια ελεγκτική εταιρεία υπογράφει την έκθεση ελέγχου της Jumbo για τη χρήση 2025;  
  gold: BDO (p. [16, 171]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2025:p67:t2, JUMBO_2025:p259:t0, JUMBO_2025:p168:t0
- `jumbo24-06` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Jumbo για τη χρήση 2024;  
  gold: BDO (p. [15, 165]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2024:p6:t0, JUMBO_2024:p17:t3, JUMBO_2024:p17:t0
- `ote25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2025;  
  gold: Deloitte (p. [199, 207]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p201:t0, OTE_2025:p148:t3, OTE_2025:p273:t2

**wrong_answer** examples:

- `ppc24-05` Σε πόσο ανήλθαν οι συνολικές επενδύσεις της ΔΕΗ το 2024;  
  gold: 3 δισ. € (p. [12]) · got: Οι συνολικές επενδύσεις του Ομίλου ΔΕΗ για το 2024 ανήλθαν σε **€ 1,875 εκατ.** [1] · retrieved: PPC_2024:p14:t1, PPC_2024:p12:t1, PPC_2024:p577:t6

**false_refusal** examples:

- `ote24-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2024;  
  gold: Deloitte (p. [205, 209, 211, 219]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2024:p211:t1, OTE_2024:p282:t10, OTE_2024:p159:t1

**ok_bad_citation** examples:

- `jumbo25-01` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Jumbo για τη χρήση 2025;  
  gold: 1.232.904.830 € (p. [179]) · got: Ο ενοποιημένος κύκλος εργασιών του Ομίλου Jumbo για τη χρήση 2025 ήταν **1.232,90 εκατ. ευρώ**【1】. · retrieved: JUMBO_2025:p18:t5, JUMBO_2025:p18:t0, JUMBO_2025:p168:t0
- `jumbo25-02` Ποια ήταν τα κέρδη μετά από φόρους του Ομίλου Jumbo το 2025;  
  gold: 320.309.417 € (p. [179, 215]) · got: Τα Καθαρά Ενοποιημένα Κέρδη μετά από φόρους του Ομίλου Jumbo το 2025 διαμορφώθηκαν σε **320,31 εκατ. ευρώ**【5】. · retrieved: JUMBO_2025:p225:t0, JUMBO_2025:p20:t0, JUMBO_2025:p215:t6
- `jumbo25-03` Ποια ήταν τα βασικά κέρδη ανά μετοχή του Ομίλου Jumbo το 2025;  
  gold: 2,3839 €/μετοχή (p. [179]) · got: Τα βασικά κέρδη ανά μετοχή του Ομίλου Jumbo για το 2025 ήταν **2,3839 € ανά μετοχή**【2】【4】. · retrieved: JUMBO_2025:p215:t4, JUMBO_2025:p215:tab2, JUMBO_2025:p3:t0

## Error analysis: bm25 · chunk600 · groq:qwen/qwen3.8-27b

| outcome | count | share | meaning |
|---|---|---|---|
| ok | 47 | 51% | Correct, and cites the right page |
| ok_bad_citation | 15 | 16% | Correct value, but no citation on the gold page |
| retrieval_miss | 13 | 14% | Gold page not in the retrieved sources |
| false_refusal | 1 | 1% | Gold page retrieved, but the model said it could not find the answer |
| wrong_answer | 1 | 1% | Gold page retrieved, but the answer is wrong (unit, year, misread table) |
| refused_correctly | 16 | 17% | Unanswerable question, correctly refused |

**retrieval_miss** examples:

- `jumbo25-06` Ποια ελεγκτική εταιρεία υπογράφει την έκθεση ελέγχου της Jumbo για τη χρήση 2025;  
  gold: BDO (p. [16, 171]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2025:p67:t2, JUMBO_2025:p259:t0, JUMBO_2025:p168:t0
- `jumbo24-06` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Jumbo για τη χρήση 2024;  
  gold: BDO (p. [15, 165]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: JUMBO_2024:p6:t0, JUMBO_2024:p17:t3, JUMBO_2024:p17:t0
- `ote25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2025;  
  gold: Deloitte (p. [199, 207]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2025:p201:t0, OTE_2025:p148:t3, OTE_2025:p273:t2

**wrong_answer** examples:

- `ppc24-05` Σε πόσο ανήλθαν οι συνολικές επενδύσεις της ΔΕΗ το 2024;  
  gold: 3 δισ. € (p. [12]) · got: Οι συνολικές επενδύσεις του Ομίλου ανήλθαν στο ποσό των € 1.875εκατ. για το έτος 2024 [1]. · retrieved: PPC_2024:p14:t1, PPC_2024:p12:t1, PPC_2024:p577:t6

**false_refusal** examples:

- `ote24-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2024;  
  gold: Deloitte (p. [205, 209, 211, 219]) · got: Δεν βρέθηκε η πληροφορία στα έγγραφα. · retrieved: OTE_2024:p211:t1, OTE_2024:p282:t10, OTE_2024:p159:t1

**ok_bad_citation** examples:

- `jumbo25-01` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Jumbo για τη χρήση 2025;  
  gold: 1.232.904.830 € (p. [179]) · got: Ο ενοποιημένος κύκλος εργασιών του Ομίλου Jumbo για τη χρήση 2025 διαμορφώθηκε σε 1.232,90 εκατ. ευρώ [1]. · retrieved: JUMBO_2025:p18:t5, JUMBO_2025:p18:t0, JUMBO_2025:p168:t0
- `jumbo25-02` Ποια ήταν τα κέρδη μετά από φόρους του Ομίλου Jumbo το 2025;  
  gold: 320.309.417 € (p. [179, 215]) · got: Τα Καθαρά Ενοποιημένα Κέρδη μετά από φόρους του Ομίλου διαμορφώθηκαν σε 320,31 εκατ. ευρώ για τη χρήση 01.01.2025-31.12.2025 [5]. · retrieved: JUMBO_2025:p225:t0, JUMBO_2025:p20:t0, JUMBO_2025:p215:t6
- `jumbo25-03` Ποια ήταν τα βασικά κέρδη ανά μετοχή του Ομίλου Jumbo το 2025;  
  gold: 2,3839 €/μετοχή (p. [179]) · got: Τα βασικά κέρδη ανά μετοχή του Ομίλου Jumbo για τη χρήση 01/01/2025-31/12/2025 ήταν 2,3839 Ευρώ ανά μετοχή [2][4]. · retrieved: JUMBO_2025:p215:t4, JUMBO_2025:p215:tab2, JUMBO_2025:p3:t0
