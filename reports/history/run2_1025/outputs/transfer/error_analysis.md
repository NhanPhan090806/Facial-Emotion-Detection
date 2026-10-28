> Historical record. Paths and model choices below describe an earlier run; use the root README for the current layout.

# transfer error analysis

Selected 10 errors out of 2363 test mistakes.
These are probability-based observations and inspection prompts; add your own visual observations.

- #1 `test/angry/PrivateTest_1054527.jpg`: Assigned sad 83.3%, versus angry 4.1% (gap 79.2%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #2 `test/disgust/PrivateTest_11895083.jpg`: Assigned fear 97.2%, versus disgust 0.3% (gap 96.9%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #3 `test/fear/PrivateTest_10254684.jpg`: Assigned angry 43.9%, versus fear 36.0% (gap 7.9%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #4 `test/happy/PrivateTest_11288161.jpg`: Assigned neutral 96.0%, versus happy 0.4% (gap 95.6%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #5 `test/neutral/PrivateTest_1129340.jpg`: Assigned sad 42.8%, versus neutral 42.1% (gap 0.7%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #6 `test/sad/PrivateTest_10658656.jpg`: Assigned surprise 93.4%, versus sad 0.2% (gap 93.1%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #7 `test/surprise/PrivateTest_10782186.jpg`: Assigned angry 65.9%, versus surprise 3.3% (gap 62.7%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #8 `test/angry/PrivateTest_11296953.jpg`: Assigned sad 47.5%, versus angry 19.5% (gap 28.0%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #9 `test/angry/PrivateTest_12000629.jpg`: Assigned surprise 58.8%, versus angry 4.5% (gap 54.3%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #10 `test/angry/PrivateTest_12191716.jpg`: Assigned neutral 46.7%, versus angry 13.1% (gap 33.6%). Inspect the shared facial cues and whether this label is visually ambiguous.

## Most frequent confusions

- sad -> neutral: 238 images.
- fear -> sad: 211 images.
- neutral -> sad: 182 images.
- sad -> fear: 142 images.
- angry -> sad: 141 images.