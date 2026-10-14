"""Port of the game's behaviour-tree library (``BehaviorTree`` namespace).

The NPCs, merchants and ravens are behaviour trees in the game; their timing
(who talks when, how long a conversation lasts, when a bird finishes fleeing)
follows from this library's exact semantics, so it is ported literally:

* every node remembers its last result and the manager tick it last ran;
* a node whose last result was not ``RUNNING`` gets ``on_new_run`` before
  running; one that was running but skipped a tick gets ``resume_run``;
* sequences and running-selectors resume at the child that was running.

Timers accumulate the 1/60 s frame delta in float32 like ``JumpKing.Timer``.
"""
import numpy as np

RUNNING, SUCCESS, FAILURE, NULL = 0, 1, 2, 3
DT = np.float32(1.0 / 60.0)


class Timer:
    """JumpKing.Timer (float32)."""

    __slots__ = ("t", "duration")

    def __init__(self, duration):
        self.duration = np.float32(max(0.0, duration))
        self.t = np.float32(0.0)

    def reset(self):
        self.t = np.float32(0.0)

    def update(self, dt=DT):
        """True when the duration has elapsed (the timer winds back)."""
        if self.duration == 0:
            return True
        self.t = np.float32(self.t + np.float32(dt))
        if self.t >= self.duration:
            while self.t >= self.duration:
                self.t = np.float32(self.t - self.duration)
            return True
        return False


class Node:
    __slots__ = ("last_result", "last_tick")

    def __init__(self):
        self.last_result = NULL
        self.last_tick = 0

    def run(self, tick):
        consecutive = tick - 1 == self.last_tick
        if self.last_result != RUNNING:
            self.on_new_run()
        if self.last_result == RUNNING and not consecutive:
            self.resume_run()
        result = self.my_run(tick)
        self.last_result = result
        self.last_tick = tick
        return result

    def my_run(self, tick):
        raise NotImplementedError

    def on_new_run(self):
        pass

    def resume_run(self):
        pass

    def reset_result(self):
        self.last_result = NULL

    @property
    def running(self):
        return self.last_result == RUNNING


class Composite(Node):
    __slots__ = ("children",)

    def __init__(self, *children):
        super().__init__()
        self.children = list(children)

    def add(self, *children):
        self.children.extend(children)
        return self

    def reset_result(self):
        super().reset_result()
        for c in self.children:
            c.reset_result()


class Sequence(Composite):
    """BTsequencor: resumes at the running child."""

    __slots__ = ("index",)

    def __init__(self, *children):
        super().__init__(*children)
        self.index = 0

    def my_run(self, tick):
        start = self.index
        self.index = 0
        for i in range(start, len(self.children)):
            r = self.children[i].run(tick)
            if r == FAILURE:
                return FAILURE
            if r == RUNNING:
                self.index = i
                return RUNNING
        return SUCCESS

    def resume_run(self):
        self.index = 0

    def reset_result(self):
        super().reset_result()
        self.index = 0


class Selector(Composite):
    """BTselector: first child that does not fail."""

    __slots__ = ()

    def my_run(self, tick):
        for c in self.children:
            r = c.run(tick)
            if r != FAILURE:
                return r
        return FAILURE


class SelectorRunning(Composite):
    """BTselectorRunning: a selector that resumes at the running child."""

    __slots__ = ("index",)

    def __init__(self, *children):
        super().__init__(*children)
        self.index = 0

    def my_run(self, tick):
        start = self.index
        self.index = 0
        for i in range(start, len(self.children)):
            r = self.children[i].run(tick)
            if r == RUNNING:
                self.index = i
                return RUNNING
            if r == SUCCESS:
                return SUCCESS
        return FAILURE

    def resume_run(self):
        self.index = 0

    def reset_result(self):
        super().reset_result()
        self.index = 0


class Simultaneous(Composite):
    """BTsimultaneous: runs every child; running wins over success over failure."""

    __slots__ = ()

    def my_run(self, tick):
        result = FAILURE
        for c in self.children:
            r = c.run(tick)
            if r == SUCCESS and result == FAILURE:
                result = r
            elif r == RUNNING:
                result = r
        return result


class Decorator(Node):
    __slots__ = ("child",)

    def __init__(self, child):
        super().__init__()
        self.child = child

    def reset_result(self):
        super().reset_result()
        self.child.reset_result()


class Invert(Decorator):
    __slots__ = ()

    def my_run(self, tick):
        r = self.child.run(tick)
        if r == RUNNING:
            return RUNNING
        return SUCCESS if r == FAILURE else FAILURE


class Repeat(Decorator):
    """BTrepeator: runs the child until it returns ``target``."""

    __slots__ = ("target",)

    def __init__(self, child, target):
        super().__init__(child)
        self.target = target

    def my_run(self, tick):
        return SUCCESS if self.child.run(tick) == self.target else RUNNING


class Static(Decorator):
    """StaticNode: runs the child, always returns ``result``."""

    __slots__ = ("result",)

    def __init__(self, child, result):
        super().__init__(child)
        self.result = result

    def my_run(self, tick):
        self.child.run(tick)
        return self.result


class Constant(Node):
    """StaticNodeSimple / EndNode (EndNode = Constant(RUNNING))."""

    __slots__ = ("result",)

    def __init__(self, result):
        super().__init__()
        self.result = result

    def my_run(self, tick):
        return self.result


class Call(Node):
    """A leaf running ``fn()`` (returns a result, or None for SUCCESS)."""

    __slots__ = ("fn",)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def my_run(self, tick):
        r = self.fn()
        return SUCCESS if r is None else r


class Check(Node):
    """A condition leaf: SUCCESS if ``fn()`` is true, FAILURE otherwise.

    Pass bound methods (not lambdas) so that trees survive ``copy.deepcopy``."""

    __slots__ = ("fn",)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def my_run(self, tick):
        return SUCCESS if self.fn() else FAILURE


check = Check


class Pause(Node):
    """PauseNode: running until ``duration`` seconds have passed."""

    __slots__ = ("timer",)

    def __init__(self, duration):
        super().__init__()
        self.timer = Timer(duration)

    def on_new_run(self):
        self.timer.reset()

    def resume_run(self):
        self.timer.reset()

    def my_run(self, tick):
        return SUCCESS if self.timer.update() else RUNNING


class Manager:
    """BTmanager: owns the tick counter of one tree."""

    __slots__ = ("root", "tick", "last_result")

    def __init__(self, root):
        self.root = root
        self.tick = 0
        self.last_result = NULL

    def run(self):
        self.tick += 1
        self.last_result = self.root.run(self.tick)
        return self.last_result
