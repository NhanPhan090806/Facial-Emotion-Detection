> Historical record. Paths and model choices below describe an earlier run; use the root README for the current layout.

# transfer error analysis

Selected 10 errors out of 2428 test mistakes.
These are probability-based observations and inspection prompts; add your own visual observations.

- #1 `test/angry/PrivateTest_1054527.jpg`: Assigned sad 98.2%, versus angry 0.7% (gap 97.5%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #2 `test/disgust/PrivateTest_11895083.jpg`: Assigned fear 75.1%, versus disgust 1.7% (gap 73.4%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #3 `test/fear/PrivateTest_10153550.jpg`: Assigned angry 67.5%, versus fear 6.4% (gap 61.1%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #4 `test/happy/PrivateTest_10516065.jpg`: Assigned surprise 78.6%, versus happy 16.7% (gap 62.0%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #5 `test/neutral/PrivateTest_11164800.jpg`: Assigned fear 60.1%, versus neutral 15.5% (gap 44.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #6 `test/sad/PrivateTest_10658656.jpg`: Assigned surprise 96.9%, versus sad 0.1% (gap 96.8%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #7 `test/surprise/PrivateTest_10427382.jpg`: Assigned happy 67.5%, versus surprise 13.5% (gap 54.0%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #8 `test/angry/PrivateTest_10590091.jpg`: Assigned fear 39.9%, versus angry 24.3% (gap 15.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #9 `test/angry/PrivateTest_12000629.jpg`: Assigned fear 52.7%, versus angry 5.5% (gap 47.2%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #10 `test/angry/PrivateTest_12191716.jpg`: Assigned happy 41.6%, versus angry 21.1% (gap 20.5%). Inspect the shared facial cues and whether this label is visually ambiguous.

## Most frequent confusions

- sad -> neutral: 237 images.
- sad -> fear: 179 images.
- fear -> sad: 163 images.
- neutral -> sad: 137 images.
- sad -> angry: 136 images.