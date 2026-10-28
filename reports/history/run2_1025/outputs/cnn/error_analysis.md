> Historical record. Paths and model choices below describe an earlier run; use the root README for the current layout.

# cnn error analysis

Selected 10 errors out of 2457 test mistakes.
These are probability-based observations and inspection prompts; add your own visual observations.

- #1 `test/angry/PrivateTest_1054527.jpg`: Assigned sad 86.6%, versus angry 2.9% (gap 83.7%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #2 `test/disgust/PrivateTest_11895083.jpg`: Assigned surprise 77.9%, versus disgust 2.3% (gap 75.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #3 `test/fear/PrivateTest_10254684.jpg`: Assigned angry 31.3%, versus fear 17.7% (gap 13.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #4 `test/happy/PrivateTest_11288161.jpg`: Assigned neutral 72.3%, versus happy 15.7% (gap 56.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #5 `test/neutral/PrivateTest_11164800.jpg`: Assigned angry 63.6%, versus neutral 15.1% (gap 48.5%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #6 `test/sad/PrivateTest_10247676.jpg`: Assigned angry 27.0%, versus sad 14.9% (gap 12.1%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #7 `test/surprise/PrivateTest_10427382.jpg`: Assigned happy 46.0%, versus surprise 34.1% (gap 11.9%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #8 `test/angry/PrivateTest_10590091.jpg`: Assigned neutral 30.5%, versus angry 24.7% (gap 5.9%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #9 `test/angry/PrivateTest_12000629.jpg`: Assigned neutral 33.0%, versus angry 20.5% (gap 12.4%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #10 `test/angry/PrivateTest_12403575.jpg`: Assigned disgust 69.6%, versus angry 17.2% (gap 52.4%). Inspect the shared facial cues and whether this label is visually ambiguous.

## Most frequent confusions

- sad -> neutral: 362 images.
- fear -> neutral: 188 images.
- fear -> sad: 178 images.
- angry -> neutral: 160 images.
- sad -> angry: 132 images.