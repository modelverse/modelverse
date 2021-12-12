import time

import numpy as np


class EarlyStopException(Exception):
    pass


class Clock:
    """ A simple clock that rings based on a given schedule """

    def __init__(self, start_iter, max_iter, scheduler='periodic_counter', period=1, warmup=None, skip_first=False,
                 udf=None):
        """ Init clock """

        # constants
        self.max_iter = max_iter  # 0 indexed
        self.start_iter = start_iter  # 0 indexed
        self.scheduler = scheduler
        self.period = period
        self.warmup = 0 if warmup is None else warmup
        self.udf = udf
        self.skip_first = skip_first

        # checks
        self._checks()

        # state
        self.start_time = time.time()
        self.last_ring_iter = self.start_iter - 1
        self.last_ring_time = self.start_time - 1
        self.curr_iter = self.start_iter - 1
        self.ring_iter_history = []
        self.ring_time_history = []
        self._is_ringing = False
        self.dead = False  # if dead, the clock is frozen and tick() and other methods have no affect
        self.state_array = np.zeros((self.max_iter - self.start_iter + 1,), dtype=int)
        self.reset()

    def _checks(self):
        assert self.warmup >= 0
        assert isinstance(self.skip_first, bool)
        if self.scheduler in ['periodic_counter']:
            assert self.period >= 1
        assert self.scheduler in ['periodic_timer', 'periodic_counter',
                                  'udf'], f"Unsupported ring scheduler {self.scheduler}"
        assert self.start_iter >= 0
        assert self.max_iter >= self.start_iter

    @property
    def is_started(self):
        return False if self.curr_iter < self.start_iter else True

    @property
    def is_terminated(self):
        if self.dead:
            return True
        elif self.current_iter == self.max_iter:
            return True
        else:
            return False

    @property
    def current_iter(self):
        return self.curr_iter

    @property
    def is_ringing(self):
        return self._is_ringing

    def tick(self):
        if self.is_terminated:
            raise StopIteration("Clock out of ticks. Max iteration reached")

        self.curr_iter += 1
        self._is_ringing = self._ring()

        if self._is_ringing:
            self.last_ring_time = time.time()
            self.last_ring_iter = self.curr_iter
            self.ring_iter_history.append(self.curr_iter)
            self.ring_time_history.append(self.last_ring_time)

        # update state array
        self.state_array[self.curr_iter - self.start_iter] = 2 if self._is_ringing else 1

    def kill(self):
        self.dead = True

    def reset(self):
        self.start_time = time.time()
        self.last_ring_iter = self.start_iter - 1
        self.last_ring_time = self.start_time - 1
        self.curr_iter = self.start_iter - 1
        self.ring_iter_history = []
        self.ring_time_history = []
        self._is_ringing = False
        self.dead = False
        self.state_array = np.zeros((self.max_iter - self.start_iter + 1,), dtype=int)

    def _udf(self):
        """ User defined function for notification """

        flag = self.udf(self.max_iter,
                        self.curr_iter,
                        self.last_ring_iter,
                        self.last_ring_time)

        return flag

    def _counter(self):
        """ Notify every k iters except the first w (warmup). """

        flag = False
        if self.curr_iter == self.max_iter:  # always notify last
            flag = True
        elif (not self.skip_first) and (self.curr_iter == self.start_iter):  # notify first
            flag = True
        elif self.skip_first and (self.curr_iter == self.start_iter):  # don't notify first
            flag = False
        elif self.curr_iter - self.start_iter < self.warmup:  # do not notify for warmup iters
            flag = False
        elif self.curr_iter == self.last_ring_iter:  # if iter hasn't updated yet
            flag = True
        elif (self.curr_iter - self.start_iter) % self.period == 0:  # notify periodically
            flag = True

        return flag

    def _timer(self):
        """ Notify every k seconds except the first w (warmup). """

        flag = False
        current_time = time.time()

        if self.curr_iter == self.max_iter:  # always notify last
            flag = True
        elif (not self.skip_first) and (self.curr_iter == self.start_iter):  # notify first
            flag = True
        elif self.skip_first and (self.curr_iter == self.start_iter):  # don't notify first
            flag = False
        elif current_time - self.last_ring_time < self.warmup:  # do not notify for warmup secs
            flag = False
        elif self.curr_iter == self.last_ring_iter:  # if iter hasn't updated yet
            flag = True
        elif current_time - self.last_ring_time >= self.period:  # notify periodically
            flag = True

        return flag

    def _ring(self):
        if self.scheduler == 'periodic_timer':
            flag = self._timer()
        elif self.scheduler == 'periodic_counter':
            flag = self._counter()
        elif self.scheduler == 'udf':
            flag = self._udf()
        else:
            raise NotImplementedError(f"Unsupported scheduler {self.scheduler}")

        return flag

    def __str__(self):
        # current state of clock as str
        w = len(str(self.max_iter))
        fmt = ""
        for ix, state in enumerate(self.state_array):
            if state == 0:
                state_str = " "
            elif state == 1:
                state_str = "-"
            elif state == 2:
                state_str = "*"
            else:
                state_str = "?"
            fmt += f"\n[{ix + self.start_iter:<{w}}] {state_str}"

        return fmt


class MultiClock:
    """ Multiple clocks with synced schedules """

    def __init__(self, mode, num_clocks, start_iter, max_iter, scheduler, period, warmup, skip_first, udf):
        """ Init Clock"""

        # constants
        self.mode = mode
        self.num_clocks = num_clocks
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.scheduler = scheduler
        self.period = period
        self.warmup = warmup
        self.udf = udf
        self.skip_first = skip_first

        # checks
        self._checks()

        # state
        self.start_time = time.time()
        self.last_ring_iter = self.start_iter - 1
        self.last_ring_time = self.start_time - 1
        self.clocks = [Clock(start_iter=self.start_iter, max_iter=self.max_iter, scheduler=self.scheduler,
                             period=self.period, warmup=self.warmup,
                             udf=self.udf, skip_first=self.skip_first) for _ in range(self.num_clocks)]
        self.curr_clock = None
        self.ring_iter_history = []
        self.ring_time_history = []
        self._is_ringing = False
        self.stopped_early = False
        self.early_stopping_iter = None
        self.state_array = np.zeros((self.max_iter - self.start_iter + 1, self.num_clocks), dtype=int)
        self.reset()

    def _checks(self):
        # parallel clock = 1,2,3,1,2,3,...
        # seq clock = 1,1,1,1,1,2,2,2,2,2,3,3,3,3,3
        assert self.mode in ['parallel', 'sequential']

    @property
    def is_started(self):
        if self.curr_clock is None:
            return False
        else:
            return self.clocks[self.curr_clock].is_started

    @property
    def is_terminated(self):
        # termination can happen only at last clock
        if self.curr_clock is None:
            return False
        elif (self.curr_clock == self.num_clocks - 1) and self.clocks[self.curr_clock].is_terminated:
            return True
        else:
            return False

    @property
    def current_iter(self):
        if self.curr_clock is None:
            return None
        else:
            return self.clocks[self.curr_clock].curr_iter

    @property
    def current_clock(self):
        return self.curr_clock

    @property
    def is_ringing(self):
        return self._is_ringing

    def _sync_seq_clocks(self):
        # in sequential mode, after first clock is terminated, the remaining clocks must ring at exactly the same iters
        # as the first clock. Hence, replace them with clocks with new schedules
        ring_iters = self.clocks[0].ring_iter_history

        def udf_schedule(max_iter, current_iter, last_ring_iter, last_ring_time):
            flag = current_iter in ring_iters
            return flag

        # replace clock schedules
        for i in range(1, self.num_clocks):
            self.clocks[i] = Clock(start_iter=self.start_iter, max_iter=self.max_iter, scheduler='udf', period=None,
                                   warmup=None, udf=udf_schedule, skip_first=self.skip_first)

    def tick(self):
        if self.is_terminated:
            raise StopIteration("Multiclock terminated. Cannot tick further.")

        # (1) select clock to tick
        if self.curr_clock is None:
            sel_clock = 0

        else:
            if self.mode == 'parallel':
                sel_clock = (self.curr_clock + 1) % self.num_clocks

            elif self.mode == 'sequential':
                # in sequential mode, when the first clock is terminated/dead, all others must be synchronized
                # to follow exact same ring schedule. Additionally all clocks must die at same iter.
                if (self.curr_clock == 0) and self.clocks[0].is_terminated:
                    self._sync_seq_clocks()

                sel_clock = (self.curr_clock + 1) % self.num_clocks if self.clocks[
                    self.curr_clock].is_terminated else self.curr_clock

            else:
                raise Exception(f"Unsupported mode {self.mode}")

        self.curr_clock = sel_clock

        # (2) tick selected clock
        self.clocks[self.curr_clock].tick()

        # (3) kill clock if stopped early
        if self.early_stopping_iter is not None:
            if self.current_iter == self.early_stopping_iter:
                self.clocks[self.curr_clock].kill()

        # (4) update ringing status
        # in parallel mode, all clocks ring or not ring together
        # async happens for example when clocks ring interval are based on time rather than count
        self._is_ringing = self.clocks[0].is_ringing if self.mode == 'parallel' else self.clocks[
            self.curr_clock].is_ringing

        # if stopped early, last iters of all clocks must be ringing
        # if (self.mode == 'sequential') and self.clocks[self.curr_clock].is_terminated:
        #     self._is_ringing = True
        if self._is_ringing:
            self.last_ring_time = time.time()
            self.last_ring_iter = (self.curr_clock, self.current_iter)
            self.ring_iter_history.append(self.last_ring_iter)
            self.ring_time_history.append(self.last_ring_time)

        # (5) update state array
        self.state_array[self.current_iter - self.start_iter, self.current_clock] = 2 if self.is_ringing else 1

    def reset(self):
        self.start_time = time.time()
        self.last_ring_iter = self.start_iter - 1
        self.last_ring_time = self.start_time - 1
        self.clocks = [Clock(start_iter=self.start_iter, max_iter=self.max_iter, scheduler=self.scheduler,
                             period=self.period, warmup=self.warmup,
                             udf=self.udf, skip_first=self.skip_first) for _ in range(self.num_clocks)]
        self.curr_clock = None
        self.ring_iter_history = []
        self.ring_time_history = []
        self._is_ringing = False
        self.stopped_early = False
        self.early_stopping_iter = None
        self.state_array = np.zeros((self.max_iter - self.start_iter + 1, self.num_clocks), dtype=int)

    def stop_early(self):
        # in parallel mode, early stop:
        # 1) can happen only at last clock
        # 2) no further ticks are possible
        if self.mode == 'parallel':
            if self.curr_clock == self.num_clocks - 1:
                self.stopped_early = True
                self.early_stopping_iter = self.current_iter
                for clock in self.clocks:  # kill all clocks
                    clock.kill()
                # self._is_ringing = True  # if stopped early, last iter must be ringing
                # self.state_array[self.current_iter - self.start_iter, self.curr_clock] = 2
            else:
                raise EarlyStopException(f"Cannot stop parallel Multiclock at current clock {self.curr_clock}. "
                                         f"For early stop, current clock must be the last clock in parallel mode")

        # in sequential model, early stop:
        # 1) can happen only at first clock
        # 2) all other clocks will also stop early at same iteration
        elif self.mode == 'sequential':
            if self.curr_clock == 0:
                self.stopped_early = True
                self.early_stopping_iter = self.current_iter
                self.clocks[0].kill()  # kill first clock
                # self._is_ringing = True  # if stopped early, last iters of all clocks must be ringing
                # self.state_array[self.current_iter - self.start_iter, 0] = 2

            else:
                raise EarlyStopException(f"Cannot stop parallel Multiclock at current clock {self.curr_clock}. "
                                         f"For early stop, current clock must be the first clock in sequential mode")

    def __str__(self):
        # current state of multiclock
        fmt = ""
        for ix, r in enumerate(self.state_array):
            line = [f"[{ix + self.start_iter:<2}]"]
            for _ in r:
                if _ == 0:
                    line.append(' ')
                elif _ == 1:
                    line.append('-')
                elif _ == 2:
                    line.append('*')
            fmt += '\n' + ' '.join(line)
        return fmt
