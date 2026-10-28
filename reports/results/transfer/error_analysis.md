# transfer error analysis

Selected 10 errors out of 2006 test mistakes.
These are probability-based observations and inspection prompts; add your own visual observations.

- #1 `test/angry/PrivateTest_1054527.jpg`: Assigned sad 87.1%, versus angry 4.7% (gap 82.3%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #2 `test/disgust/PrivateTest_34426922.jpg`: Assigned fear 33.5%, versus disgust 18.4% (gap 15.1%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #3 `test/fear/PrivateTest_10555537.jpg`: Assigned angry 70.4%, versus fear 8.4% (gap 61.9%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #4 `test/happy/PrivateTest_11288161.jpg`: Assigned neutral 83.3%, versus happy 14.5% (gap 68.8%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #5 `test/neutral/PrivateTest_11164800.jpg`: Assigned angry 79.2%, versus neutral 7.2% (gap 72.1%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #6 `test/sad/PrivateTest_10247676.jpg`: Assigned fear 46.5%, versus sad 21.0% (gap 25.5%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #7 `test/surprise/PrivateTest_10089743.jpg`: Assigned fear 70.8%, versus surprise 28.4% (gap 42.4%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #8 `test/angry/PrivateTest_12000629.jpg`: Assigned neutral 80.1%, versus angry 7.8% (gap 72.4%). Confident error; inspect label ambiguity, face crop and domain mismatch.
- #9 `test/angry/PrivateTest_12403575.jpg`: Assigned disgust 66.9%, versus angry 8.7% (gap 58.2%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #10 `test/angry/PrivateTest_12766285.jpg`: Assigned happy 49.2%, versus angry 4.4% (gap 44.8%). Inspect the shared facial cues and whether this label is visually ambiguous.

## Most frequent confusions

- sad -> neutral: 200 images.
- fear -> sad: 169 images.
- sad -> fear: 159 images.
- fear -> angry: 135 images.
- neutral -> sad: 131 images.