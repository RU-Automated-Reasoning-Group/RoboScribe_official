import pdb
from synthesize.dsl import Program, Exists, Box, GoalDistance
from collections import deque
import numpy as np
import pickle
import time

# one input format
# { "target": Box, "all_box": [Box], "result": True/False }


def check_program(all_inputs, program):
    correct = 0
    for idx, input in enumerate(all_inputs):
        # print(input)
        rst = program.evaluate_specific(input)
        # print(rst)
        if rst == input["result"]:
            correct += 1
    print("accuracy is", correct / len(all_inputs), f"({correct}/{len(all_inputs)})")
    return correct/len(all_inputs)

def topdown(all_inputs, val):
    q = deque()
    idx = 0
    q.append([idx, Exists([], Program([], val))])
    idx += 1
    import pdb
    t_start = time.time()
    x = Exists([], Program([], val))
    # x = x.expand()[2]
    # x = x.expand()[4]
    # x = x.expand()[1]
    # x = x.expand()[0]
    # x = x.expand()[2]
    # x = x.expand()[8]
    # x = x.expand()[11]
    # x = x.expand()[14]
    # q.append([0, x])
    # pdb.set_trace()
    while len(q):
        n, p = q.popleft()
        print(f"checking {n} program", p)
        if n % 10000 == 0:
            print("time from start", time.time() - t_start)
        if p.is_complete():
            print("complete")
            acc = check_program(all_inputs, p)
            if acc > 0.95:
                return p
        else:            
            if idx > 100:
                # only check the first 1 million programs for now
                continue
            ps = p.expand()
            ps_with_idx = []

            for new_p in ps:
                ps_with_idx.append([idx, new_p])
                idx += 1
            print([(idx, str(p)) for idx, p in ps_with_idx])
            q.extend(ps_with_idx)
        # break
    return None


def goaldis(obj):
    loc = np.array([obj.x, obj.y, obj.z])
    goal_loc = np.array([obj.gx, obj.gy, obj.gz])
    return np.sqrt(np.sum((loc - goal_loc) ** 2))


if __name__ == "__main__":
    # with open("demo_img.pkl", "rb") as f:
    #     demo_imgs = pickle.load(f)
    # with open("demo_act.pkl", "rb") as f:
    #     demo_act = pickle.load(f)
    with open("demo_obs.pkl", "rb") as f:
        demo_obs = pickle.load(f)

    # pdb.set_trace()
    all_inputs = []
    for obs_seq in demo_obs:
        for idx, obs in enumerate(obs_seq):
            boxes = []
            for i in range(3):
                b = Box(str(i))
                b.set_attribute(
                    obs[3 * i + 5],
                    obs[3 * i + 6],
                    obs[3 * i + 7],
                    obs[3 * i + 19],
                    obs[3 * i + 20],
                    obs[3 * i + 21],
                )
                boxes.append(b)
            all_inputs.append(
                {"target": boxes[2], "all_box": boxes, "result": idx <= 18}
            )
            all_inputs.append(
                {"target": boxes[1], "all_box": boxes, "result": idx > 18}
            )
            print(
                idx, goaldis(boxes[2]), goaldis(boxes[1]), goaldis(boxes[0]), idx <= 18
            )

        break
    # pdb.set_trace()
    # exit()
    topdown(all_inputs)
