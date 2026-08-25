package natskv

// Startup-observability tests: every StartMember/StartClient step that
// performs I/O must be bounded by the provider-owned StartStepTimeout, and a
// step failure must name the step, the bucket, the key, and the elapsed time
// -- never a bare "context deadline exceeded". They run against in-process
// fakes of jetstream.KeyValue / jetstream.JetStream, because the failure
// under test is a server that accepts the connection and then never answers,
// which an embedded healthy server cannot reproduce.

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nats-io/nats.go/jetstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/awevoke/protoactor-go/actor"
	"github.com/awevoke/protoactor-go/cluster"
	"github.com/awevoke/protoactor-go/remote"
)

// startObsGuard is how long a test waits before declaring a liveness
// expectation broken: in runStepBounded, for a step the provider is supposed
// to bound to return; in respinWindow, for the watch-loop goroutine to exit
// after shutdown. It is far above every StartStepTimeout the tests configure,
// so a pass is never a lucky race.
const startObsGuard = 5 * time.Second

// startObsStepTimeout is the provider-owned per-step bound the tests
// configure: long enough that fake bookkeeping never trips it accidentally,
// short enough that the suite stays fast.
const startObsStepTimeout = 200 * time.Millisecond

// stepFakeWatcher is a hand-rolled jetstream.KeyWatcher: the test owns the
// updates channel and observes Stop.
type stepFakeWatcher struct {
	updates chan jetstream.KeyValueEntry
	stopped atomic.Bool
}

func (w *stepFakeWatcher) Updates() <-chan jetstream.KeyValueEntry { return w.updates }

func (w *stepFakeWatcher) Stop() error {
	w.stopped.Store(true)
	return nil
}

// stepFakeEntry is a jetstream.KeyValueEntry carrying just what the member
// drain reads: key, value, operation.
type stepFakeEntry struct {
	key   string
	value []byte
	op    jetstream.KeyValueOp
}

func (e stepFakeEntry) Bucket() string                  { return "" }
func (e stepFakeEntry) Key() string                     { return e.key }
func (e stepFakeEntry) Value() []byte                   { return e.value }
func (e stepFakeEntry) Revision() uint64                { return 1 }
func (e stepFakeEntry) Created() time.Time              { return time.Time{} }
func (e stepFakeEntry) Delta() uint64                   { return 0 }
func (e stepFakeEntry) Operation() jetstream.KeyValueOp { return e.op }

// stepFakeKV is a jetstream.KeyValue whose Put/Create/Watch the test scripts.
// The embedded interface is nil: any method the code under test reaches that
// the test did not script panics, which is exactly the alarm we want.
type stepFakeKV struct {
	jetstream.KeyValue
	putFn      func(ctx context.Context, key string, value []byte) (uint64, error)
	createFn   func(ctx context.Context, key string, value []byte) (uint64, error)
	watchFn    func(ctx context.Context, keys string, opts ...jetstream.WatchOpt) (jetstream.KeyWatcher, error)
	watchCalls atomic.Int64
}

func (f *stepFakeKV) Put(ctx context.Context, key string, value []byte) (uint64, error) {
	if f.putFn != nil {
		return f.putFn(ctx, key, value)
	}
	return 1, nil
}

func (f *stepFakeKV) Create(ctx context.Context, key string, value []byte, _ ...jetstream.KVCreateOpt) (uint64, error) {
	if f.createFn != nil {
		return f.createFn(ctx, key, value)
	}
	return 1, nil
}

func (f *stepFakeKV) Watch(ctx context.Context, keys string, opts ...jetstream.WatchOpt) (jetstream.KeyWatcher, error) {
	f.watchCalls.Add(1)
	if f.watchFn != nil {
		return f.watchFn(ctx, keys, opts...)
	}
	return &stepFakeWatcher{updates: make(chan jetstream.KeyValueEntry)}, nil
}

// blockUntilCtxDone is the Put/Create shape of the incident: a server that
// accepts the call and never answers. It returns only when the caller's
// context does, with the context's error -- the same thing the real client
// surfaces when its PubAck never arrives.
func blockUntilCtxDone(ctx context.Context, _ string, _ []byte) (uint64, error) {
	<-ctx.Done()
	return 0, ctx.Err()
}

// stepFakeJetStream hands out scripted KVs by bucket name from
// CreateOrUpdateKeyValue. The embedded interface is nil, as in stepFakeKV.
type stepFakeJetStream struct {
	jetstream.JetStream
	buckets map[string]*stepFakeKV
	// createBucketFn, when set, replaces the map lookup entirely.
	createBucketFn func(ctx context.Context, cfg jetstream.KeyValueConfig) (jetstream.KeyValue, error)
}

func (f *stepFakeJetStream) CreateOrUpdateKeyValue(ctx context.Context, cfg jetstream.KeyValueConfig) (jetstream.KeyValue, error) {
	if f.createBucketFn != nil {
		return f.createBucketFn(ctx, cfg)
	}
	kv, ok := f.buckets[cfg.Bucket]
	if !ok {
		return nil, fmt.Errorf("stepFakeJetStream: unscripted bucket %q", cfg.Bucket)
	}
	return kv, nil
}

// newStepTestProvider builds a Provider around an empty stepFakeJetStream --
// it scripts no buckets; callers assign p.memberBucket / p.leaderBucket
// directly -- with the context wiring StartMember would have done, ready for
// direct step calls.
func newStepTestProvider(t *testing.T, opts ...Option) *Provider {
	t.Helper()

	p, err := NewFromJetStream(&stepFakeJetStream{}, opts...)
	require.NoError(t, err)

	p.clusterName = "obscluster"
	p.ctx, p.cancel = context.WithCancel(context.Background())
	t.Cleanup(p.cancel)

	p.self = NewNode("obscluster_self", "127.0.0.1", 4222, nil)
	p.isMember = true

	return p
}

// runStepBounded runs fn on its own goroutine and fails the test if it does
// not return within startObsGuard -- the exact symptom of an unbounded step.
func runStepBounded(t *testing.T, what string, fn func() error) (time.Duration, error) {
	t.Helper()

	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- fn() }()

	select {
	case err := <-done:
		return time.Since(start), err
	case <-time.After(startObsGuard):
		t.Fatalf("%s did not return within %v; the provider-owned step timeout is not bounding it", what, startObsGuard)
		return 0, nil
	}
}

// TestStartMember_BlockingRegisterSelfFailsWithinStepTimeout is the incident
// in miniature: buckets ensure fine, then the register-self Put never gets
// its PubAck. StartMember must fail within the provider-owned StartStepTimeout
// -- it must not run to the client's own implicit ~5s API default and fail
// with an error that names only the step -- and the error must name the step, the
// bucket, the key, and the elapsed time, because "register self: context
// deadline exceeded" has twice cost hours of triage.
func TestStartMember_BlockingRegisterSelfFailsWithinStepTimeout(t *testing.T) {
	memberKV := &stepFakeKV{putFn: blockUntilCtxDone}
	leaderKV := &stepFakeKV{}

	js := &stepFakeJetStream{buckets: map[string]*stepFakeKV{
		"protoactor_obscluster_members":        memberKV,
		"protoactor_obscluster_members_leader": leaderKV,
	}}

	p, err := NewFromJetStream(js, WithStartStepTimeout(startObsStepTimeout))
	require.NoError(t, err)

	system := actor.NewActorSystem()
	remoteConfig := remote.Configure("127.0.0.1", 0)
	clusterConfig := cluster.Configure("obscluster", p, p.IdentityLookup(), remoteConfig)
	c := cluster.NewCluster(system, clusterConfig)
	c.Remote = remote.NewRemote(system, remoteConfig)

	elapsed, err := runStepBounded(t, "StartMember", func() error { return p.StartMember(c) })
	t.Cleanup(p.cancel)

	require.Error(t, err)
	assert.Less(t, elapsed, 2*time.Second,
		"StartMember must fail within StartStepTimeout plus scheduling slack, got %v", elapsed)

	msg := err.Error()
	assert.Contains(t, msg, "register self", "the failing step must be named: %v", err)
	assert.Contains(t, msg, "protoactor_obscluster_members", "the bucket must be named: %v", err)
	assert.Contains(t, msg, p.memberKey(p.self.ID), "the key must be named: %v", err)
	assert.Contains(t, msg, "elapsed", "the elapsed time must be reported: %v", err)
}

// TestRegisterSelf_StepTimeoutReachesTheClientCall pins HOW the bound is
// applied: the context handed to the KV Put must carry a deadline, so the
// client library's own hidden default (5s on a deadline-less context) never
// decides when a startup step fails.
func TestRegisterSelf_StepTimeoutReachesTheClientCall(t *testing.T) {
	sawDeadline := make(chan bool, 1)
	memberKV := &stepFakeKV{putFn: func(ctx context.Context, key string, value []byte) (uint64, error) {
		_, ok := ctx.Deadline()
		sawDeadline <- ok
		<-ctx.Done()
		return 0, ctx.Err()
	}}

	p := newStepTestProvider(t, WithStartStepTimeout(startObsStepTimeout))
	p.memberBucket = memberKV

	_, err := runStepBounded(t, "registerSelf", p.registerSelf)
	require.Error(t, err)
	assert.True(t, <-sawDeadline,
		"the Put context must carry the provider-owned deadline; a deadline-less context leaves the bound to the client's hidden default")
}

// TestBucketEnsureErrors_NameBucketAndElapsed covers the two bucket-ensure
// steps: their failures already named the bucket, and must now also report
// elapsed time, so a slow-then-failed ensure is distinguishable from a fast
// rejection.
func TestBucketEnsureErrors_NameBucketAndElapsed(t *testing.T) {
	sentinel := errors.New("nats: no responders")

	for _, tc := range []struct {
		name   string
		step   func(p *Provider) error
		phrase string
		bucket string
	}{
		{
			name:   "member bucket",
			step:   func(p *Provider) error { return p.createMemberBucket() },
			phrase: "create member bucket",
			bucket: "protoactor_obscluster_members",
		},
		{
			name:   "leader bucket",
			step:   func(p *Provider) error { return p.createLeaderBucket() },
			phrase: "create leader bucket",
			bucket: "protoactor_obscluster_members_leader",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newStepTestProvider(t, WithStartStepTimeout(startObsStepTimeout))
			p.js = &stepFakeJetStream{createBucketFn: func(_ context.Context, _ jetstream.KeyValueConfig) (jetstream.KeyValue, error) {
				return nil, sentinel
			}}

			err := tc.step(p)
			require.Error(t, err)
			require.ErrorIs(t, err, sentinel, "the cause must stay wrapped: %v", err)

			msg := err.Error()
			assert.Contains(t, msg, tc.phrase, "the step must be named: %v", err)
			assert.Contains(t, msg, tc.bucket, "the bucket must be named: %v", err)
			assert.Contains(t, msg, "elapsed", "the elapsed time must be reported: %v", err)
		})
	}
}

// TestLoadInitialMembers_StalledDrainFailsWithinStepTimeout covers the
// initial member load step's DRAIN half -- the piece with no client bound.
// Establishment itself is capped ~5s by the client even on a deadline-less
// context (the legacy subscribe path under kv.Watch wraps it with the legacy
// JS context's MaxWait) and is not exercised here: the fake watchFn returns
// an established watcher immediately. What must hold: a watcher that never
// delivers its initial values fails the drain within StartStepTimeout with
// the named detail, and the established watcher is stopped on the way out.
func TestLoadInitialMembers_StalledDrainFailsWithinStepTimeout(t *testing.T) {
	watcher := &stepFakeWatcher{updates: make(chan jetstream.KeyValueEntry)} // never delivers
	memberKV := &stepFakeKV{watchFn: func(_ context.Context, _ string, _ ...jetstream.WatchOpt) (jetstream.KeyWatcher, error) {
		return watcher, nil
	}}

	p := newStepTestProvider(t, WithStartStepTimeout(startObsStepTimeout))
	p.memberBucket = memberKV

	elapsed, err := runStepBounded(t, "loadInitialMembers", p.loadInitialMembers)
	require.Error(t, err)
	assert.Less(t, elapsed, 2*time.Second,
		"loadInitialMembers must fail within StartStepTimeout plus scheduling slack, got %v", elapsed)

	msg := err.Error()
	assert.Contains(t, msg, "load initial members", "the step must be named: %v", err)
	assert.Contains(t, msg, "protoactor_obscluster_members", "the bucket must be named: %v", err)
	assert.Contains(t, msg, "cluster.members.>", "the watched keys must be named: %v", err)
	assert.Contains(t, msg, "elapsed", "the elapsed time must be reported: %v", err)
	assert.True(t, watcher.stopped.Load(), "the abandoned watcher must be stopped, or its consumer leaks")
}

// TestLoadInitialMembers_DrainCompletionUnchanged pins the success path the
// timeout must not disturb: history entries are applied, delete markers are
// skipped, and the nil sentinel ends the drain with no error.
func TestLoadInitialMembers_DrainCompletionUnchanged(t *testing.T) {
	p := newStepTestProvider(t, WithStartStepTimeout(startObsStepTimeout))

	nodeA := NewNode("obscluster_a", "127.0.0.1", 4223, nil)
	dataA, err := nodeA.Serialize()
	require.NoError(t, err)
	nodeB := NewNode("obscluster_b", "127.0.0.1", 4224, nil)
	dataB, err := nodeB.Serialize()
	require.NoError(t, err)
	nodeGone := NewNode("obscluster_gone", "127.0.0.1", 4225, nil)
	dataGone, err := nodeGone.Serialize()
	require.NoError(t, err)

	updates := make(chan jetstream.KeyValueEntry, 4)
	updates <- stepFakeEntry{key: p.memberKey(nodeA.ID), value: dataA, op: jetstream.KeyValuePut}
	// The delete marker carries a real serialized node on purpose: a drain
	// that stopped skipping delete markers would decode it and add a third
	// member, so the Len==2 assertion below genuinely pins the skip. A
	// value-less delete entry would not -- NewNodeFromBytes(nil) fails and
	// the non-skipping path passes silently.
	updates <- stepFakeEntry{key: p.memberKey(nodeGone.ID), value: dataGone, op: jetstream.KeyValueDelete}
	updates <- stepFakeEntry{key: p.memberKey(nodeB.ID), value: dataB, op: jetstream.KeyValuePut}
	updates <- nil // end-of-initial-values sentinel

	p.memberBucket = &stepFakeKV{watchFn: func(_ context.Context, _ string, _ ...jetstream.WatchOpt) (jetstream.KeyWatcher, error) {
		return &stepFakeWatcher{updates: updates}, nil
	}}

	_, err = runStepBounded(t, "loadInitialMembers", p.loadInitialMembers)
	require.NoError(t, err)

	p.membersMu.RLock()
	defer p.membersMu.RUnlock()
	assert.Len(t, p.members, 2)
	assert.Contains(t, p.members, nodeA.ID)
	assert.Contains(t, p.members, nodeB.ID)
}

// TestAttemptLeaderElection_CreateBoundedByStepTimeout covers the leader
// election join: its Create is deliberately fire-and-forget (any error means
// "someone else leads"), but the CALL itself must still be bounded so that a
// wedged server cannot park StartMember's tail -- or a background re-election
// -- forever.
func TestAttemptLeaderElection_CreateBoundedByStepTimeout(t *testing.T) {
	leaderKV := &stepFakeKV{createFn: blockUntilCtxDone}

	p := newStepTestProvider(t, WithStartStepTimeout(startObsStepTimeout))
	p.leaderBucket = leaderKV

	elapsed, _ := runStepBounded(t, "attemptLeaderElection", func() error {
		p.attemptLeaderElection()
		return nil
	})

	assert.Less(t, elapsed, 2*time.Second,
		"attemptLeaderElection must return within StartStepTimeout plus scheduling slack, got %v", elapsed)
	assert.False(t, p.IsLeader(), "a timed-out Create must not confer leadership")
}

// respinWindow observes a watch loop against a bucket whose watcher closes
// cleanly (nil error) the moment it is established -- the server-side shape
// that previously produced a zero-delay respin loop, i.e. a consumer-creation
// hot loop. It returns how many Watch calls the loop made in the window.
func respinWindow(t *testing.T, p *Provider, kv *stepFakeKV, start func(), window time.Duration) int64 {
	t.Helper()

	start()
	time.Sleep(window)

	p.shutdown.Store(true)
	p.cancel()

	waitDone := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(waitDone)
	}()
	select {
	case <-waitDone:
	case <-time.After(startObsGuard):
		t.Fatal("watch loop did not exit after shutdown")
	}

	return kv.watchCalls.Load()
}

// closedWatchKV returns a stepFakeKV whose every watcher is already closed:
// keepWatching / keepWatchingLeader see an immediately-ended Updates stream
// and return nil.
func closedWatchKV() *stepFakeKV {
	return &stepFakeKV{watchFn: func(_ context.Context, _ string, _ ...jetstream.WatchOpt) (jetstream.KeyWatcher, error) {
		updates := make(chan jetstream.KeyValueEntry)
		close(updates)
		return &stepFakeWatcher{updates: updates}, nil
	}}
}

// TestStartWatching_CleanCloseRespinHonorsRetryInterval pins that a clean
// watcher close waits RetryInterval before the next Watch, exactly as an
// erroring close always has. Before the fix the nil-return path respun with
// zero delay: against a 300ms window and a 50ms interval that is a measured
// ~1.9M Watch calls (consumer creations) instead of at most a handful.
func TestStartWatching_CleanCloseRespinHonorsRetryInterval(t *testing.T) {
	kv := closedWatchKV()
	p := newStepTestProvider(t, WithRetryInterval(50*time.Millisecond))
	p.memberBucket = kv

	calls := respinWindow(t, p, kv, p.startWatching, 300*time.Millisecond)

	assert.GreaterOrEqual(t, calls, int64(2), "the loop must keep respinning")
	assert.LessOrEqual(t, calls, int64(10),
		"a clean close must wait RetryInterval before respinning; %d Watch calls in 300ms is a hot loop", calls)
}

// TestStartLeaderWatching_CleanCloseRespinHonorsRetryInterval is the same pin
// for the leader-key watch loop, which has the identical respin shape.
func TestStartLeaderWatching_CleanCloseRespinHonorsRetryInterval(t *testing.T) {
	kv := closedWatchKV()
	p := newStepTestProvider(t, WithRetryInterval(50*time.Millisecond))
	p.leaderBucket = kv

	calls := respinWindow(t, p, kv, p.startLeaderWatching, 300*time.Millisecond)

	assert.GreaterOrEqual(t, calls, int64(2), "the loop must keep respinning")
	assert.LessOrEqual(t, calls, int64(10),
		"a clean close must wait RetryInterval before respinning; %d Watch calls in 300ms is a hot loop", calls)
}

// TestStartStepTimeoutConfig pins the default and the option.
func TestStartStepTimeoutConfig(t *testing.T) {
	cfg := newDefaultConfig()
	assert.Equal(t, defaultStartStepTimeout, cfg.StartStepTimeout)
	assert.Equal(t, 10*time.Second, defaultStartStepTimeout,
		"the default must exceed the nats.go implicit 5s per-call default it replaces")

	WithStartStepTimeout(3 * time.Second)(cfg)
	assert.Equal(t, 3*time.Second, cfg.StartStepTimeout)

	// <= 0 disables the provider-owned bound (steps fall back to whatever the
	// client library applies), mirroring ReconcileInterval's convention.
	WithStartStepTimeout(0)(cfg)
	assert.Equal(t, time.Duration(0), cfg.StartStepTimeout)
}
