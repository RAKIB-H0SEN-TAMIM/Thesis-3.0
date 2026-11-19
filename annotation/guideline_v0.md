# Annotation guideline v0 (SentNoB sentiment + audit verdict)

Status: **draft v0, 19 Nov 2026**. Week 3-এ 100-item pilot-এর পর v1 হবে। প্রতিটা পরিবর্তন নিচের changelog-এ লেখো, কারণ paper-এ guideline version report করতে হবে।

Source: SentNoB paper (Islam et al., Findings of EMNLP 2021)-এর label definition। **Week 1-এ Researcher A paper-এর annotation section পড়ে নিচের definition মিলিয়ে নেবে**; কোথাও না মিললে original paper-এর definition জিতবে, আর এখানে লিখে রাখবে।

---

## ১. Step 1: Blind labeling (gold দেখা যাবে না)

প্রতিটা comment পড়ো, নিজের label দাও। Gold label, অন্যজনের sheet, বা detector-এর output কিছুই দেখবে না। Item নিয়ে partner-এর সাথে আলোচনা করবে না।

| Label | কখন দেবে | উদাহরণ (ধরন) |
|---|---|---|
| **Positive** | প্রশংসা, সমর্থন, খুশি, ধন্যবাদ, ভালো লাগা প্রকাশ | "অসাধারণ ভিডিও ভাই", "খুব ভাল লাগলো" |
| **Negative** | সমালোচনা, রাগ, গালি, দুঃখ, হতাশা, অভিযোগ | "এত অপরিচ্ছন্ন পরিবেশের খাবার খাওয়া ঠিক না" |
| **Neutral** | Factual statement, প্রশ্ন, request, তথ্য, বা স্পষ্ট কোনো অনুভূতি নেই | "ভাইয়া নতুন ভিডিও আসে না কেন" |
| **Can't decide** | শুধু তখনই, যখন সত্যিই দুইটা label সমান defensible বা text বোঝা যায় না | |

### Decision rules (boundary case)

1. **লেখকের নিজের অনুভূতি** label করো, বিষয়ের না। "দুর্ঘটনায় ৫ জন মারা গেছে" (খবর) = Neutral; "এই দুর্ঘটনা দেখে কষ্ট লাগছে" = Negative।
2. **Mixed sentiment:** comment-এর মূল বক্তব্য (সাধারণত শেষ অংশ বা সবচেয়ে জোরালো অংশ) যেদিকে, সেই label। সত্যিই 50/50 হলে Can't decide, আর `note`-এ "mixed" লেখো।
3. **Sarcasm/irony:** আক্ষরিক অর্থ না, বোঝানো অর্থ label করো ("বাহ, কি সুন্দর রাস্তা, গাড়ি চালানোই যায় না" = Negative)। `sarcasm` column-এ y দাও।
4. **Request/প্রশ্ন:** সাধারণত Neutral। কিন্তু request-এর ভেতর স্পষ্ট প্রশংসা/অভিযোগ থাকলে সেটা জিতবে ("ভাই আপনার ভিডিও দারুণ, আরো দিন" = Positive)।
5. **দোয়া/ধর্মীয় formula** ("আল্লাহ আপনার মঙ্গল করুক", "ইনশাআল্লাহ"): কাউকে উদ্দেশ্য করে শুভকামনা হলে Positive; শুধু অভ্যাসগত phrase হলে বাকি বক্তব্য দেখে ঠিক করো।
6. **নিজের channel promote / spam** ("আমার চ্যানেলটা সাবস্ক্রাইব করুন"): Neutral।
7. **Dialect / আঞ্চলিক শব্দ** বা **Romanized/English-mixed** text: অর্থ বুঝে label দাও, আর `dialect` / `code_mixed_or_romanized` column-এ y দাও। এই tag-গুলো দিয়েই আমরা দেখব LLM detector code-mixed/dialect text-এ খারাপ করে কিনা (cross-check fix b)।
8. Confidence: 1 = আন্দাজ, 2 = মোটামুটি নিশ্চিত, 3 = পুরো নিশ্চিত।

---

## ২. Step 2: Adjudication (এখন gold দেখা যাবে)

শুধু সেই item, যেখানে অন্তত একজন gold-এর সাথে একমত না। দুজনে মিলে (পরে external annotator / supervisor সহ) প্রতিটার verdict দাও:

| Verdict | মানে |
|---|---|
| **Correct** | Gold ঠিক আছে; step 1-এ আমাদের ভুল ছিল |
| **Wrong** | Gold স্পষ্টভাবে ভুল। `correct_label` আর `taxonomy` দাও |
| **Ambiguous** | দুইটা label-ই যুক্তিসঙ্গত। `taxonomy` দাও |
| **Unintelligible** | Text বোঝা যায় না |

**Wrong** তখনই দেবে, যখন একজন যুক্তিবাদী native speaker guideline পড়ে gold label-কে defend করতে পারবে না। সন্দেহ থাকলে Ambiguous। এই নিয়মটাই আমাদের error rate-কে conservative রাখবে, আর reviewer-এর "তোমরা নিজেরাই over-flag করেছ" আপত্তির উত্তর এটাই।

---

## ৩. Bangla-specific error taxonomy (v0)

Wrong আর Ambiguous item-এ একটা code দাও। এটাই paper-এর contribution (a)।

| Code | Category | কী দেখে বুঝবে |
|---|---|---|
| E1 | Clear slip | স্পষ্ট ভুল click; text-এ কোনো জটিলতা নেই |
| E2 | Sarcasm / irony | আক্ষরিক অর্থে label দেওয়া হয়েছে |
| E3 | Dialect word | আঞ্চলিক শব্দ ভুল বোঝা হয়েছে |
| E4 | Code-mixed / romanized | English/Banglish অংশ ভুল পড়া হয়েছে |
| E5 | Neutral boundary | হালকা অনুভূতিকে Neutral (বা উল্টো) |
| E6 | Mixed sentiment | Positive + Negative দুটোই আছে |
| E7 | Target confusion | ঘটনার প্রতি Negative কিন্তু ব্যক্তির প্রতি supportive, ইত্যাদি |
| E8 | Religious / formulaic phrase | দোয়া, ইনশাআল্লাহ, সালাম ইত্যাদির inconsistent label |
| E9 | Question / request | প্রশ্ন বা request-কে sentiment দেওয়া |
| E10 | Needs outside context | ভিডিও/post না দেখে বোঝা যায় না |
| E11 | Duplicate label conflict | একই text dataset-এ অন্য label নিয়েও আছে |
| E12 | Other | `note`-এ লেখো; ৫টার বেশি হলে নতুন code বানাও |

Pilot-এর পর যে code ১%-এর কম আসে সেটা merge করো, আর বারবার "Other" এলে নতুন code বানাও। VITD আর BLP-2025 hate dataset-এর জন্য v1-এ আলাদা section লাগবে (implicit hate, slur vs. reclaimed use, political vs. religious target)।

---

## ৪. Annotator logistics

- Session ২৫ মিনিটের বেশি না; hate/violence dataset-এ আরো ছোট। মন খারাপ লাগলে থামো।
- ~১০ সেকেন্ড/item (blind step), ~৩০ সেকেন্ড/item (adjudication)।
- External blind annotator (cross-check fix 2): Week 3-এর মধ্যে অন্তত একজন বাইরের native speaker (অন্য department-এর বন্ধু, ছোট honorarium বা acknowledgement)। তাকে শুধু Step 1 sheet দেওয়া হবে, detector source ছাড়া।

## Changelog
- v0 (19 Nov 2026): first draft from SentNoB README + blueprint section G. Not yet checked against the paper.
