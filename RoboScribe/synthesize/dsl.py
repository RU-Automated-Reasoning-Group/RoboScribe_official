import string
import copy
from typing import List
import numpy as np

id_num = 0


# def get_num():
# global id_num
# old_num = id_num
# id_num += 1
# return string.ascii_lowercase[old_num]


def get_num(ctx):
    n = len(ctx)
    return string.ascii_lowercase[n]


class gx:
    def __call__(self, obj):
        return obj.gx

    def __str__(self) -> str:
        return "gx"


class gy:
    def __call__(self, obj):
        return obj.gy

    def __str__(self) -> str:
        return "gy"


class gz:
    def __call__(self, obj):
        return obj.gz

    def __str__(self) -> str:
        return "gz"


class EnvObj:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id


class Box(EnvObj):
    def __init__(self, id):
        super().__init__(id)
        self.funcs = [gx(), gy(), gz()]
        self.set = False

    def set_attribute(self, x, y, z, gx, gy, gz):
        self.x = x
        self.y = y
        self.z = z
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.set = True

    def __str__(self):
        if self.set:
            return f"Box{self.id}: ({self.x}, {self.y}, {self.z}, {self.gx}, {self.gy}, {self.gz})"
        return f"{self.id}"


class Program:
    def __init__(self, ctx: List[EnvObj], predicate_val):
        self.ctx = copy.deepcopy(ctx)
        self.predicate_val = predicate_val

    def is_complete(self):
        return False

    def evaluate(self, input):
        raise NotImplemented

    def expand(self):
        # return a list of programs
        ps = [
            Exists(self.ctx, Program(self.ctx, self.predicate_val)),
            Not(self.ctx, Program(self.ctx, self.predicate_val)),
            And(self.ctx, Program(self.ctx, self.predicate_val), Program(self.ctx, self.predicate_val)),
            Or(self.ctx, Program(self.ctx, self.predicate_val), Program(self.ctx, self.predicate_val)),
        ]
        ps.extend(Predicate(self.ctx, self.predicate_val).expand())
        return ps

    def __str__(self):
        return "p"

    def add_object(self, obj):
        self.ctx.append(obj)


class Exists(Program):
    def __init__(self, ctx: List[EnvObj], program: Program):
        import pdb

        # pdb.set_trace()
        self.ctx = copy.deepcopy(ctx)
        self.program = copy.deepcopy(program)
        self.object = Box(get_num(self.ctx))
        self.program.add_object(self.object)

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        # check existence for all object
        for obj in input["all_box"]:
            # if self.program.evaluate(input, mapping | {self.object.id: obj}):
            if self.program.evaluate(input, {**mapping, **{self.object.id: obj}}):
                return True
        return False

    def evaluate_specific(self, input):
        # check existence using target object
        return self.program.evaluate(input, {self.object.id: input["target"]})

    def expand(self):
        sub_ps = self.program.expand()
        ps = []
        for sub_p in sub_ps:
            e = copy.deepcopy(self)
            e.program = sub_p
            ps.append(e)
        return ps

    def __str__(self):
        return f"Exists {self.object}(" + str(self.program) + ")"


class Not(Program):
    def __init__(self, ctx: List[EnvObj], program: Program):
        self.ctx = copy.deepcopy(ctx)
        self.program = copy.deepcopy(program)

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        return not self.program.evaluate(input, mapping)

    def expand(self):
        sub_ps = self.program.expand()
        return [Not(self.ctx, p) for p in sub_ps]

    def __str__(self):
        return "Not(" + str(self.program) + ")"


class And(Program):
    def __init__(self, ctx: List[EnvObj], p1: Program, p2: Program):
        self.ctx = copy.deepcopy(ctx)
        self.program1 = copy.deepcopy(p1)
        self.program2 = copy.deepcopy(p2)

    def is_complete(self):
        return self.program1.is_complete() and self.program2.is_complete()

    def evaluate(self, input, mapping):
        return self.program1.evaluate(input, mapping) and self.program2.evaluate(
            input, mapping
        )

    def expand(self):
        if not self.program1.is_complete():
            sub_ps = self.program1.expand()
            return [And(self.ctx, p, self.program2) for p in sub_ps]

        sub_ps = self.program2.expand()
        return [And(self.ctx, self.program1, p) for p in sub_ps]

    def __str__(self):
        return "And(" + str(self.program1) + ", " + str(self.program2) + ")"


class Or(Program):
    def __init__(self, ctx: List[EnvObj], p1: Program, p2: Program):
        self.ctx = copy.deepcopy(ctx)
        self.program1 = copy.deepcopy(p1)
        self.program2 = copy.deepcopy(p2)

    def is_complete(self):
        return self.program1.is_complete() and self.program2.is_complete()

    def evaluate(self, input, mapping):
        return self.program1.evaluate(input, mapping) or self.program2.evaluate(
            input, mapping
        )

    def expand(self):
        if not self.program1.is_complete():
            sub_ps = self.program1.expand()
            return [Or(self.ctx, p, self.program2) for p in sub_ps]

        sub_ps = self.program2.expand()
        return [Or(self.ctx, self.program1, p) for p in sub_ps]

    def __str__(self):
        return "Or(" + str(self.program1) + ", " + str(self.program2) + ")"


class Predicate:
    def __init__(self, ctx: List[EnvObj], val):
        self.ctx = copy.deepcopy(ctx)
        self.val = val

    def is_complete(self):
        return False

    def evaluate(self, input):
        raise NotImplemented

    def expand(self):
        programs = []
        for i in range(0, len(self.ctx)):
            for j in range(i + 1, len(self.ctx)):
                for f in self.ctx[i].funcs:
                    programs += [
                        Greater(f, f, self.ctx[i], self.ctx[j]),
                        # Equal(f, f, self.ctx[i], self.ctx[j]),
                        Less(f, f, self.ctx[i], self.ctx[j]),
                    ]
        for i in range(0, len(self.ctx)):
            programs += [GoalDistance(self.ctx[i], self.val)]
        return programs

    def add_object(self, object):
        return

    def __str__(self):
        return "Predicate"

# 0.1923
# 0.032
# 0.182
# 0.0487
# 0.012
# 0.027 (most recent)
class GoalDistance(Predicate):
    def __init__(self, obj, val):
        self.obj = obj
        self.val = val

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        obj = mapping[self.obj.id]
        return (
            np.sqrt(
                np.sum(
                    (
                        np.array([obj.x, obj.y, obj.z])
                        - np.array([obj.gx, obj.gy, obj.gz])
                    )
                    ** 2
                )
            )
            > self.val
        )

    def expand(self):
        return []

    def __str__(self):
        return f"( goalDistance({self.obj.id}) > {self.val})"


class Greater(Predicate):
    def __init__(self, func1, func2, obj1, obj2):
        import pdb

        # pdb.set_trace()
        self.func1 = func1
        self.func2 = func2
        self.obj1 = obj1
        self.obj2 = obj2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        obj1 = mapping[self.obj1.id]
        obj2 = mapping[self.obj2.id]
        return self.func1(obj1) > self.func2(obj2)

    def expand(self):
        return []

    def __str__(self):
        return (
            f"( {str(self.func1)}({self.obj1.id}) > {str(self.func2)}({self.obj2.id}))"
        )


class Equal(Predicate):
    def __init__(self, func1, func2, obj1, obj2):
        self.func1 = func1
        self.func2 = func2
        self.obj1 = obj1
        self.obj2 = obj2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        obj1 = mapping[self.obj1.id]
        obj2 = mapping[self.obj2.id]
        return self.func1(obj1) == self.func2(obj2)

    def expand(self):
        return []

    def __str__(self):
        return (
            f"( {str(self.func1)}({self.obj1.id}) = {str(self.func2)}({self.obj2.id}))"
        )


class Less(Predicate):
    def __init__(self, func1, func2, obj1, obj2):
        self.func1 = func1
        self.func2 = func2
        self.obj1 = obj1
        self.obj2 = obj2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        obj1 = mapping[self.obj1.id]
        obj2 = mapping[self.obj2.id]
        return self.func1(obj1) < self.func2(obj2)

    def expand(self):
        return []

    def __str__(self):
        return (
            f"( {str(self.func1)}({self.obj1.id}) < {str(self.func2)}({self.obj2.id}))"
        )
