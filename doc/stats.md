# Statistics

The `stats` classes and functions allow us to collect statistics about
the internal of the learning / optimization procedure. There are three
main goals for this code, which unfortunately are at odds with one 
another:
* **Non-intrusiveness**: The statistics gathering code should require as 
  little modifications to the actual computation code as possible.
* **Speed**: While statistics gathering will come with some unavoidable
  overhead, we want to have it as fast as possible. Especially if it has
  been decided at runtime that a statistics is not required, the stats
  code should have no discernible effect on running time.
* **Versatility**: Since we don't know beforehand which statistics are
  needed for an experiment, we want it so that this can be configured at
  runtime. A change in the desired statistics should (usually) require no
  modifications to the code and no recompiling.
  
### Recording implementation
In order to achieve speedy data gathering, we need to do the statistics
collection in each thread separately and independently, and only combine
data at the end. This means that instances of the types which get
replicated per-thread, and which are expected to gather statistics (e.g.
the minimizer and objective, initial condition and post processors) 
contain a statistics gatherer. While this requires adding an additional
data member to the classes, it is less intrusive than requiring to pass
some statistics gatherer as an argument of any function which might want
to collect statistics. This also allows each of these classes to define
numerical indices (needed for speed) for the different statistics
that are independent of each other, so there is no coupling between
the different statistics gathering modules. 

For the end user, the statistics need to be accessible by a 
(human-readable) name, but in the actual code a string-based lookup would
induce a performance penalty. Therefore, we require that each statistics
be declared before use in a way that assigned a unique integer id to the
name. The recording code itself can then use this ID and the statistics
can be found by a (quick) array lookup. Since each thread-local object 
(Minimizer, Objective, ...) which gathers statistics has its own 
StatisticsCollection instance, the IDs only need to be unique within each
type.

To reduce the necessary boilerplate code a base class `dismec::stats::Tracked` is provided
from which the statistics gathering objects can be derived. This wrapper is coded
in such a way that it does not actually include the `dismec::stats::StatisticsCollection`
as a header dependency (non-intrusiveness), but still allows all performance-critical
calls that will be relayed to the collection to be inlined. This is achieved by
making these calls dependent on a dummy template parameter, so a definition of 
`dismec::stats::StatisticsCollection` only needs to be present at the call site.

### Tags
In many cases we want to be able to track ancillary information attached to the main
data that is recorded, e.g. in which iteration of the optimization the recording 
happened. This is managed by tags. Contrary to the recorded statistics, tags are
persistent values that remain until overwritten. They are also associated with a 
name (for human setup / output) and an ID (for fast updating). 

Internally, a tag is
implemented as a `std::shared_ptr` to its data. In this way, any statistics that wants
to correlate its data with the tag can request (based on the name) this pointer at the
beginning of the setup phase, and during recording only has to dereference the pointer
to read the value. The reason why we don't use the same system (shared ptr that is
written to) also to update the tags from user code is that this would require each 
recording class to store these shared pointers, i.e. the possible tags would leak into
the class header, which we want to avoid.

A particularly tricky situation arises if the data we want to use as a tag is not
available to the object that records the statistics. This happens for example if we 
want to correlate a statistics in the objective function with the iteration in the
optimizer. That means the tag is not registered in the same object as the statistics
that uses it. For that reason, we can also register *read-only* tags to collections,
which are tags of other collections which can be read by the statistics of this
collection.
