# cnn error analysis

Selected 10 errors out of 2411 test mistakes.
These are probability-based observations and inspection prompts; add your own visual observations.

- #1 `test/angry/PrivateTest_1054527.jpg`: Assigned sad 79.1%, versus angry 4.5% (gap 74.6%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #2 `test/disgust/PrivateTest_11895083.jpg`: Assigned surprise 61.3%, versus disgust 2.3% (gap 59.0%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #3 `test/fear/PrivateTest_10254684.jpg`: Assigned sad 60.8%, versus fear 10.1% (gap 50.7%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #4 `test/happy/PrivateTest_11155116.jpg`: Assigned angry 41.2%, versus happy 15.4% (gap 25.8%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #5 `test/neutral/PrivateTest_11164800.jpg`: Assigned angry 67.9%, versus neutral 10.5% (gap 57.4%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #6 `test/sad/PrivateTest_10247676.jpg`: Assigned fear 31.5%, versus sad 22.3% (gap 9.2%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #7 `test/surprise/PrivateTest_10427382.jpg`: Assigned happy 63.5%, versus surprise 23.7% (gap 39.9%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #8 `test/angry/PrivateTest_11296953.jpg`: Assigned fear 42.6%, versus angry 33.9% (gap 8.7%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #9 `test/angry/PrivateTest_12000629.jpg`: Assigned neutral 33.3%, versus angry 22.2% (gap 11.2%). Inspect the shared facial cues and whether this label is visually ambiguous.
- #10 `test/angry/PrivateTest_12191716.jpg`: Assigned sad 41.5%, versus angry 26.2% (gap 15.4%). Inspect the shared facial cues and whether this label is visually ambiguous.

## Most frequent confusions

- sad -> neutral: 316 images.
- fear -> sad: 197 images.
- fear -> neutral: 165 images.
- angry -> neutral: 142 images.
- neutral -> sad: 139 images.