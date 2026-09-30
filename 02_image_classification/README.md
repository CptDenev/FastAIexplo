# Room recognition by fine-tuning

## Problem
Recognize which room of a building a photo was taken in, for a security and access use case in a public venue. Constraint: the environment cannot be modified or marked (no QR codes, no signage), so recognition has to rely on the room itself.

Built as a practical case for a job interview.

## Data
Photos taken on site at the Conciergerie (Paris), 5 rooms.

## Try it
The [HF Space](https://huggingface.co/spaces/CptDenev/Conciergerie) accepts any uploaded photo: pictures of the 5 rooms, but also out-of-distribution images to see the doubt and rejection behavior in action.

## Model
ResNet18 pretrained on ImageNet, fine-tuned with fastai. 45 MB, deployed on [HF Space](https://huggingface.co/spaces/CptDenev/Conciergerie) with a Gradio interface.

## Decision rule
The model always outputs a probability for each room. Instead of adding an "unknown" class, the answer is read from the confidence of the top prediction:

| Top probability | Output |
|---|---|
| > 80% | Room identified |
| 40 to 80% | Doubt, confirmation needed |
| Spread across several rooms | Rejected |

## Results
- Test set of 200 photos: correct room as top prediction in 100% of cases.
- Out-of-distribution photos, evaluated separately (floor only, bare wall, isolated detail such as a close-up of a lamp): no room gets a strongly dominant probability, so these cases fall into doubt or rejection instead of producing a confident wrong answer.

## Limits and next steps
- Softmax probabilities are known to be overconfident. Calibration (temperature scaling) would make the 40% and 80% thresholds more reliable.
- An explicit "not a known room" class, trained on out-of-distribution photos, would be the next comparison.

## Files
- `room_classifier.py`: dataset cleaning and fine-tuning
- `app.py`: Gradio inference app