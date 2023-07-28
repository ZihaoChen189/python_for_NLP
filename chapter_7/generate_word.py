import numpy as np

class softmax:
    pass
class RnnLM:
    pass
class betterRnnLM:
    pass


class RnnLMGen(RnnLM):
    def generate(self, start_id, skip_ids=None, sample_size=1000):
        words_ids = [start_id]
        x = start_id
        while len(words_ids) < sample_size:  # check the limit of request
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())  # OK, make the probability

            sampled = np.random.choice(len(p), size=1, p=p)  # randomly select one sample
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                words_ids.append(int(x))

        return words_ids
    