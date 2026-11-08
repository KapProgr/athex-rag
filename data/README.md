# Data

10 annual financial reports (Ετήσια Οικονομική Έκθεση, Greek edition), 5 companies × 2 years,
about 4,100 pages in total. The PDFs are not committed (they are large and owned by the
companies). Download them from the links below into `data/raw/` with these file names:

| File | Company | Sector | Pages | Source |
|---|---|---|---|---|
| `OTE_2025.pdf` | ΟΤΕ | Telecom | 300 | [telekom.gr](https://www.telekom.gr/otegroup_company/investor_relations/financial_results/financial_statements_ote_group_k_ote_sa/gr/ANNUAL_FINANCIAL_REPORT_2025_without_L3556_gr.pdf) |
| `OTE_2024.pdf` | ΟΤΕ | Telecom | 310 | [telekom.gr](https://www.telekom.gr/otegroup_company/investor_relations/financial_results/financial_statements_ote_group_k_ote_sa/gr/Annual_Financial_Report_2024_without_L3556_gr.pdf) |
| `JUMBO_2025.pdf` | Jumbo | Retail | 259 | [e-jumbo.gr](https://corporate.e-jumbo.gr/uploads/159286/44f6_jumboannualreport_31122025_gr.pdf) |
| `JUMBO_2024.pdf` | Jumbo | Retail | 246 | [e-jumbo.gr](https://corporate.e-jumbo.gr/uploads/159220/9504_jumboannualreport_31122024_gr.pdf) |
| `PPC_2025.pdf` | ΔΕΗ | Energy | 664 | [ppcgroup.com](https://www.ppcgroup.com/media/0vljitb1/ethsia-oikonomikh-ekthesh-2025.pdf) |
| `PPC_2024.pdf` | ΔΕΗ | Energy | 643 | [ppcgroup.com](https://www.ppcgroup.com/media/s0rjks5s/ethsia-oikonomikh-ekthesh-2024.pdf) |
| `MOH_2025.pdf` | Motor Oil | Refining | 207 | [moh.gr](https://www.moh.gr/wp-content/uploads/2026/05/ΕΤΗΣΙΑ-ΟΙΚΟΝΟΜΙΚΗ-ΕΚΘΕΣΗ-FY2025_ΜΟΗ_GR_WEB.pdf) |
| `MOH_2024.pdf` | Motor Oil | Refining | 243 | [moh.gr](https://www.moh.gr/wp-content/uploads/2025/06/Ετήσια-Οικονομική-Έκθεση-2024.pdf) |
| `ETE_2025.pdf` | Εθνική Τράπεζα | Banking | 668 | [nbg.gr](https://www.nbg.gr/-/jssmedia/Files/Group/enhmerwsh-ependutwn/Annual_Financial_Reports/Annual-Financial-Report-2025-GR.pdf) |
| `ETE_2024.pdf` | Εθνική Τράπεζα | Banking | 643 | [nbg.gr](https://www.nbg.gr/-/jssmedia/Files/Group/enhmerwsh-ependutwn/Annual_Financial_Reports/Annual-Financial-Report-2024-GR.pdf) |

Links were valid in November 2026; companies reorganise their sites, so if one breaks,
look for "Ετήσια Οικονομική Έκθεση" in the company's investor-relations section.

Notes on the documents (they matter for the results):

- Since 2021 the *official* filing is ESEF (inline XHTML). These PDFs are the companies'
  print versions of the same report.
- Motor Oil's pages are two-column; plain text extraction interleaves the columns.
- The OTE PDFs contain placeholder "Lorem ipsum" text in the shareholder-letter page.
- OTE restated its 2024 figures in the 2025 report after selling Telekom Romania Mobile, so
  "revenue 2024" is 3.590,8 in the 2024 report and 3.334,0 in the 2025 one. The questions
  always ask about the report's own figures.
- Motor Oil changed statutory auditor from Deloitte to KPMG for 2024.

`data/companies.json` maps each code to the names people type in questions.
`data/cache/` (created on first ingest) holds parsed pages so later runs skip PDF parsing.
