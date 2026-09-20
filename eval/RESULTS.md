# Evaluation results

Test set: 28 letters + 3 unclear clips, one speaker, letter names, single laptop microphone.

| Approach | Correct | Accuracy |
| --- | --- | --- |
| CTC candidate scoring | 24/28 | 85.7% |
| Whisper + text match | 17/28 | 60.7% |

## Per-clip

| File | Expected | CTC heard | Conf | Whisper text | Whisper matched |
| --- | --- | --- | --- | --- | --- |
| alif.m4a | alif | alif | 1.00 | ألف | alif |
| baa.m4a | baa | baa | 0.92 | بات | baa |
| taa.m4a | taa | taa | 0.96 | تأ | taa |
| thaa.m4a | thaa | baa | 0.82 | بأ | baa |
| jeem.m4a | jeem | jeem | 0.99 | جيم | jeem |
| haae.m4a | Haa | Haa | 0.93 | ها | haa |
| khaa.m4a | khaa | Taa | 0.62 | اع | alif |
| daal.m4a | daal | daal | 0.99 | دان | daal |
| thaal.m4a | dhaal | dhaal | 0.95 | ذال | dhaal |
| raa.m4a | raa | raa | 0.96 | رأى | raa |
| zaae.m4a | zay | zay | 0.72 | زيب | jeem |
| seen.m4a | seen | seen | 0.99 | سين | seen |
| sheen.m4a | sheen | sheen | 0.98 | شين | sheen |
| saad.m4a | saad | saad | 0.92 | ساد | saad |
| daad.m4a | Daad | Daad | 0.91 | داد | daal |
| taa'.m4a | Taa | Taa | 0.95 | باق | baa |
| dhaa.m4a | DHaa | DHaa | 0.40 | و | noon |
| ayn.m4a | ayn | ayn | 0.99 | اين | seen |
| ghayn.m4a | ghayn | ayn | 0.98 | وضعين | ayn |
| meem.m4a | meem | meem | 0.98 | ميم | meem |
| faa.m4a | faa | faa | 0.95 | فأ | faa |
| qaaf.m4a | qaaf | qaaf | 0.99 | قاف | qaaf |
| kaef.m4a | kaaf | kaaf | 0.99 | كيف | kaaf |
| laam.m4a | laam | laam | 1.00 | لام | laam |
| noon.m4a | noon | noon | 1.00 | نون | noon |
| haa.m4a | haa | Haa | 0.86 | ها | haa |
| waw.m4a | waw | waw | 0.96 | wow | alif |
| yae.m4a | yaa | yaa | 0.96 | yeah | alif |
