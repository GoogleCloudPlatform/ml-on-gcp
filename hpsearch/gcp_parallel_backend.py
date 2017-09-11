from sklearn.externals.joblib._parallel_backends import ParallelBackendBase, ThreadingBackend, ImmediateResult
from sklearn.externals.joblib.parallel import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

class GCEManagerMixin(object):
    pass

class GCEBackend(ParallelBackendBase, GCEManagerMixin):
    supports_timeout = False

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs that can actually run in parallel
        n_jobs is the number of workers requested by the callers. Passing
        n_jobs=-1 means requesting all available workers for instance matching
        the number of CPU cores on the worker host(s).
        This method should return a guesstimate of the number of workers that
        can actually perform work concurrently. The primary use case is to make
        it possible for the caller to know in how many chunks to slice the
        work.
        In general working on larger data chunks is more efficient (less
        scheduling overhead and better use of CPU cache prefetching heuristics)
        as long as all the workers have enough work to do.
        """
        return 1

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        # the returned value must implement a get() method
        # which must return a list of outputs
        # e.g. [[{'score': 1.0}, {'score': 1.0}, 11, 0.022326946258544922, 0.00036215782165527344]]
        # https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/model_selection/_search.py#L641
        # (train_score_dicts, test_score_dicts, test_sample_counts, fit_time, score_time)
        result = ImmediateResult(func)

        result.results = [[{'score': 1.0}, {'score': 0.5}, 11, 0.99, 0.1]]

        return result

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        """Reconfigure the backend and return the number of workers.
        This makes it possible to reuse an existing backend instance for
        successive independent calls to Parallel with different parameters.
        """

        
        
        self.parallel = parallel
        return self.effective_n_jobs(n_jobs)

    def terminate(self):
        """Shutdown the process or thread pool"""

    def compute_batch_size(self):
        """Determine the optimal batch size"""
        return 1

    def batch_completed(self, batch_size, duration):
        """Callback indicate how long it took to run a batch"""
        # might not need to be implemented?
        raise

    def get_exceptions(self):
        """List of exception types to be captured."""
        # might not need to be implemented?
        raise
        return []

    def abort_everything(self, ensure_ready=True):
        """Abort any running tasks
        This is called when an exception has been raised when executing a tasks
        and all the remaining tasks will be ignored and can therefore be
        aborted to spare computation resources.
        If ensure_ready is True, the backend should be left in an operating
        state as future tasks might be re-submitted via that same backend
        instance.
        If ensure_ready is False, the implementer of this method can decide
        to leave the backend in a closed / terminated state as no new task
        are expected to be submitted to this backend.
        Setting ensure_ready to False is an optimization that can be leveraged
        when aborting tasks via killing processes from a local process pool
        managed by the backend it-self: if we expect no new tasks, there is no
        point in re-creating a new working pool.
        """
        # Does nothing by default: to be overridden in subclasses when canceling
        # tasks is possible.
        # might not need to be implemented?
        raise
        pass

########


param_grid = {
    'learning_rate': [0.1, 0.5, 1.0, 2.0],
    'n_estimators': [100, 150, 200]
}

X = [[1, 2], [1, 3], [2, 3]] * 10
y = [1, 0, 1] * 10

gbc = GradientBoostingClassifier()

grid_search = GridSearchCV(gbc, param_grid)

gce_backend = GCEBackend()

#threading_backend = ThreadingBackend()

with parallel_backend(gce_backend):
    grid_search.fit(X, y)


