# BM25 retrieval baseline (no embeddings, no LLM)

Measured on the full evaluation set (93 questions, 77 answerable) over the 10 reports.
Retrieval only: is a gold page among the top k=5 chunks? Dense/hybrid retrieval and answer
quality need the embedding models and Ollama, which `scripts/run_all.ps1` runs; see
`experiments.md` for those once it has been run.

Each answerable question is worth about 1.3 points, so differences of 1–3 points are noise.

## Chunk size × table chunks × Greek stemming (k = 5)

| config | hit@k | mrr | hit@k_table_q | latency_s |
|---|---|---|---|---|
| chunk 600 · tables · no stem | 64.9% | 0.458 | 46.4% | 0.09 |
| chunk 600 · tables · stem6 | 71.4% | 0.528 | 46.4% | 0.09 |
| chunk 600 · no tables · no stem | 67.5% | 0.464 | 46.4% | 0.08 |
| chunk 600 · no tables · stem6 | 68.8% | 0.524 | 39.3% | 0.08 |
| chunk 1200 · tables · no stem | 63.6% | 0.498 | 50.0% | 0.06 |
| chunk 1200 · tables · stem6 | 68.8% | 0.523 | 50.0% | 0.06 |
| chunk 1200 · no tables · no stem | 67.5% | 0.502 | 60.7% | 0.04 |
| chunk 1200 · no tables · stem6 | 70.1% | 0.529 | 53.6% | 0.05 |
| chunk 2000 · tables · no stem | 64.9% | 0.479 | 53.6% | 0.03 |
| chunk 2000 · tables · stem6 | 68.8% | 0.518 | 50.0% | 0.04 |
| chunk 2000 · no tables · no stem | 67.5% | 0.481 | 60.7% | 0.03 |
| chunk 2000 · no tables · stem6 | 70.1% | 0.531 | 53.6% | 0.03 |

## Stemming prefix length (chunk 1200, with tables, k = 5)

Prefix truncation is the stemmer: tokens are cut to their first N characters
(`ελεγκτική`, `ελεγκτές` → `ελεγκτ`).

| config | hit@k | mrr | hit@k_table_q | latency_s |
|---|---|---|---|---|
| prefix – | 63.6% | 0.498 | 50.0% | 0.05 |
| prefix 4 | 64.9% | 0.480 | 46.4% | 0.05 |
| prefix 5 | 66.2% | 0.484 | 46.4% | 0.05 |
| prefix 6 | 68.8% | 0.523 | 50.0% | 0.06 |
| prefix 7 | 67.5% | 0.499 | 53.6% | 0.05 |
| prefix 8 | 68.8% | 0.479 | 57.1% | 0.05 |

## Observations

- **Stemming helps.** Cutting Greek words to 6 characters raises hit@5 (+1 to +6.5 points)
  and MRR in all six configurations. Hit on table questions does drop in two of them. The misses it fixes are inflection mismatches between question and report
  (`ελεγκτική εταιρεία` vs `Ορκωτοί Ελεγκτές`, `εργαζόμενους` vs `εργαζομένους`).
  Prefix 6 has the best MRR and is the default.
- **Table chunks do not help BM25.** A table's page text is already indexed. The extra
  table chunk mostly competes with it. Table chunks exist for the LLM, which needs clean
  rows to read a figure, so their value shows up in answer accuracy, not retrieval.
- **Typical misses:** generic financial terms (`κύκλος εργασιών`, `κέρδη μετά από φόρους`)
  appear on dozens of pages of each 200–650 page report (notes, segment tables, prior-year
  columns), so the statement page is outranked. These are the cases that dense retrieval
  and the reranker are expected to fix.

## Misses (chunk 1200 · tables · stem6 · k = 5)

| outcome | count | share | meaning |
|---|---|---|---|
| retrieval_miss | 24 | 31% | Gold page not in the retrieved sources |
| retrieved | 53 | 69% | Retrieval-only run: gold page retrieved |

**retrieval_miss** examples:

- `jumbo25-01` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Jumbo για τη χρήση 2025;  
  gold: 1.232.904.830 € (p. [179]) · got: (retrieval only) · retrieved: JUMBO_2025:p18:t3, JUMBO_2025:p168:t0, JUMBO_2025:p18:t0
- `jumbo25-03` Ποια ήταν τα βασικά κέρδη ανά μετοχή του Ομίλου Jumbo το 2025;  
  gold: 2,3839 €/μετοχή (p. [179]) · got: (retrieval only) · retrieved: JUMBO_2025:p215:t2, JUMBO_2025:p215:tab2, JUMBO_2025:p20:t2
- `jumbo25-05` Πόσους εργαζόμενους είχε η θυγατρική της Jumbo στη Ρουμανία στο τέλος του 2025;  
  gold: 1.750 άτομα (p. [36, 249]) · got: (retrieval only) · retrieved: JUMBO_2025:p35:t1, JUMBO_2025:p27:t0, JUMBO_2025:p95:t0
- `jumbo25-06` Ποια ελεγκτική εταιρεία υπογράφει την έκθεση ελέγχου της Jumbo για τη χρήση 2025;  
  gold: BDO (p. [16, 171]) · got: (retrieval only) · retrieved: JUMBO_2025:p67:t2, JUMBO_2025:p7:t0, JUMBO_2025:p259:t0
- `jumbo24-03` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών της Jumbo για τη χρήση 2024;  
  gold: 1.149.873.264 € (p. [172]) · got: (retrieval only) · retrieved: JUMBO_2024:p17:t0, JUMBO_2024:p17:t3, JUMBO_2024:p162:t0
- `jumbo24-06` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Jumbo για τη χρήση 2024;  
  gold: BDO (p. [15, 165]) · got: (retrieval only) · retrieved: JUMBO_2024:p6:t0, JUMBO_2024:p17:t2, JUMBO_2024:p235:t0
- `jumbo24-07` Με πόσα καταστήματα είχε παρουσία η Jumbo μέσω συνεργασιών (franchise) το 2024;  
  gold: 39 καταστήματα (p. [18]) · got: (retrieval only) · retrieved: JUMBO_2024:p234:t1, JUMBO_2024:p142:t1, JUMBO_2024:p31:t4
- `ote25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις του ΟΤΕ για το 2025;  
  gold: Deloitte (p. [199, 207]) · got: (retrieval only) · retrieved: OTE_2025:p201:t0, OTE_2025:p148:t2, OTE_2025:p273:t0
- `ote25-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2025;  
  gold: 3.464,3 εκατ. € (p. [190, 212]) · got: (retrieval only) · retrieved: OTE_2025:p54:t3, OTE_2025:p89:t1, OTE_2025:p139:t4
- `ote25-03` Ποια ήταν τα κέρδη χρήσης του Ομίλου ΟΤΕ για το 2025;  
  gold: 501,1 εκατ. € (p. [212, 213]) · got: (retrieval only) · retrieved: OTE_2025:p256:t1, OTE_2025:p164:t2, OTE_2025:p280:t0
- `ote25-04` Πόσες ήταν οι ζημιές από διακοπείσες δραστηριότητες του Ομίλου ΟΤΕ το 2025;  
  gold: 224,9 εκατ. € (p. [212]) · got: (retrieval only) · retrieved: OTE_2025:p181:t0, OTE_2025:p55:t2, OTE_2025:p38:t3
- `ote25-06` Ποια θυγατρική πούλησε ο ΟΤΕ την 1 Οκτωβρίου 2025;  
  gold: Telekom Romania Mobile (p. [19]) · got: (retrieval only) · retrieved: OTE_2025:p20:t4, OTE_2025:p20:t3, OTE_2025:p255:t1
- `ote24-02` Ποιος ήταν ο συνολικός κύκλος εργασιών του Ομίλου ΟΤΕ το 2024, όπως δημοσιεύθηκε στην ετήσια έκθεση 2024;  
  gold: 3.590,8 εκατ. € (p. [200, 224]) · got: (retrieval only) · retrieved: OTE_2024:p65:t1, OTE_2024:p64:t3, OTE_2024:p89:t2
- `ppc25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της ΔΕΗ για το 2025;  
  gold: Ernst & Young (p. [650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664]) · got: (retrieval only) · retrieved: PPC_2025:p649:t0, PPC_2025:p431:t2, PPC_2025:p432:t2
- `ppc25-07` Σε πόσο ανήλθε το προσαρμοσμένο EBITDA της ΔΕΗ το 2025;  
  gold: 2 δισ. € (p. [13]) · got: (retrieval only) · retrieved: PPC_2025:p11:t0, PPC_2025:p437:t3, PPC_2025:p437:t2
- `moh25-01` Ποια ελεγκτική εταιρεία έλεγξε τις οικονομικές καταστάσεις της Motor Oil για το 2025;  
  gold: KPMG (p. [204, 206]) · got: (retrieval only) · retrieved: MOH_2025:p200:t1, MOH_2025:p14:t2, MOH_2025:p170:t4
- `moh25-02` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Motor Oil το 2025;  
  gold: 11.482.478 χιλ. € (p. [4]) · got: (retrieval only) · retrieved: MOH_2025:p52:t2, MOH_2025:p125:t5, MOH_2025:p163:t3
- `moh25-03` Ποια ήταν τα καθαρά κέρδη μετά από φόρους του Ομίλου Motor Oil το 2025;  
  gold: 650.807 χιλ. € (p. [4, 142]) · got: (retrieval only) · retrieved: MOH_2025:p27:tab3, MOH_2025:p27:t2, MOH_2025:p9:t1
- `moh25-06` Πόσο προσωπικό απασχολούσε ο Όμιλος ΗΛΕΚΤΩΡ στις 31.12.2025;  
  gold: 725 άτομα (p. [17]) · got: (retrieval only) · retrieved: JUMBO_2025:p188:t0, JUMBO_2025:p36:t0, OTE_2025:p217:t2
- `moh25-07` Ποιο ήταν το EBITDA της μητρικής εταιρείας Motor Oil (σε επίπεδο Εταιρείας) το 2025;  
  gold: 745.876 χιλ. € (p. [4]) · got: (retrieval only) · retrieved: MOH_2025:p125:t5, MOH_2025:p170:t4, MOH_2025:p14:t2
- `moh24-02` Ποια ήταν τα καθαρά κέρδη μετά από φόρους του Ομίλου Motor Oil το 2024;  
  gold: 287.314 χιλ. € (p. [4, 178]) · got: (retrieval only) · retrieved: MOH_2024:p102:t1, MOH_2024:p30:tab1, MOH_2024:p35:t0
- `moh24-03` Ποιος ήταν ο ενοποιημένος κύκλος εργασιών του Ομίλου Motor Oil το 2024;  
  gold: 12.187.917 χιλ. € (p. [4]) · got: (retrieval only) · retrieved: MOH_2024:p72:t0, MOH_2024:p122:t5, MOH_2024:p42:t2
- `ete25-03` Ποιος ήταν ο δείκτης MREL της Εθνικής Τράπεζας στο τέλος του 2025;  
  gold: 29,2% (p. [50]) · got: (retrieval only) · retrieved: ETE_2025:p572:t5, ETE_2025:p54:t3, ETE_2025:p572:t2
- `ete25-04` Ποιος ήταν ο δείκτης Κόστος προς Έσοδα της Εθνικής Τράπεζας το 2025;  
  gold: 34,1% (p. [50]) · got: (retrieval only) · retrieved: ETE_2025:p49:t3, ETE_2025:p448:t1, ETE_2025:p448:tab0
