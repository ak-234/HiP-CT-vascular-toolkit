"""Scale-equivariant constrained multiscale centreline smoothing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve

from .config import runtime_config as config
from .implicit_field import CapsuleBVH

SCALES = (1.0, 2.0, 4.0)
KAPPA_R_MAX = 0.95

#: Per-segment backtracking rounds in the certified line search. Each round
#: halves the step of every segment still in conflict, so the worst case decays
#: as 2**-rounds; 24 reaches the floor below from 1.0.
_BACKTRACK_ROUNDS = 24
#: Step fractions below this are snapped to zero -- a segment that still
#: conflicts at 0.1% of its correction is not going to be rescued by a smaller
#: one, and reverting it outright frees the remaining rounds for others.
_ALPHA_FLOOR = 1.0e-3


@dataclass(frozen=True)
class SmoothingReport:
    converged: bool
    iterations: int
    lambda_min: float
    lambda_median: float
    lambda_max: float
    max_displacement_radius: float
    p95_displacement_radius: float
    curvature_violations_before: int
    curvature_violations_after: int
    self_distance_violations_before: int
    self_distance_violations_after: int
    input_overlaps: int
    new_branch_conflicts: int
    frozen_points: int
    unresolved_constraints: int
    backtrack_fraction: float
    modified_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Seg:
    sid: int
    n1: int
    n2: int
    idx: np.ndarray
    r: np.ndarray
    xi: np.ndarray


@dataclass
class _Edge:
    seg: int
    local: int
    a: int
    b: int
    r0: float
    r1: float
    s0: float
    s1: float
    length: float


@dataclass
class _Conflict:
    ei: int
    ej: int
    si: int
    sj: int
    u: float
    v: float
    clearance: float

    @property
    def key(self):
        return (min(self.ei, self.ej), max(self.ei, self.ej))


def _xi(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x))
    out[1:] = np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1) /
                        np.maximum(.5 * (r[:-1] + r[1:]), 1e-15))
    return out


def _weights(xi: np.ndarray) -> np.ndarray:
    if len(xi) < 2:
        return np.ones(len(xi))
    d = np.maximum(np.diff(xi), 1e-12)
    w = np.empty(len(xi)); w[0] = d[0] / 2; w[-1] = d[-1] / 2
    if len(xi) > 2: w[1:-1] = (d[:-1] + d[1:]) / 2
    return w


def _second_difference(xi: np.ndarray, r: np.ndarray, h: float) -> sparse.csr_matrix:
    rr=[]; cc=[]; vv=[]; row=0; w=_weights(xi)
    for i in range(1, len(xi)-1):
        a = min(int(np.searchsorted(xi, xi[i]-h, side="right")-1), i-1)
        b = max(int(np.searchsorted(xi, xi[i]+h, side="left")), i+1)
        if a < 0 or b >= len(xi): continue
        ha, hb = xi[i]-xi[a], xi[b]-xi[i]
        if min(ha, hb) <= 1e-12: continue
        q = np.sqrt(max(w[i],1e-12))/max(r[i],1e-12)*2/(ha+hb)
        for j,c in ((a,q/ha),(i,-q*(1/ha+1/hb)),(b,q/hb)):
            rr.append(row); cc.append(j); vv.append(c)
        row += 1
    return sparse.csr_matrix((vv,(rr,cc)), shape=(row,len(xi)))


def _noise(x: np.ndarray, xi: np.ndarray, r: np.ndarray) -> float:
    z=[]
    for i in range(1,len(x)-1):
        a=min(int(np.searchsorted(xi,xi[i]-1,side="right")-1),i-1)
        b=max(int(np.searchsorted(xi,xi[i]+1,side="left")),i+1)
        if a < 0 or b >= len(x) or xi[b] <= xi[a]: continue
        t=(xi[i]-xi[a])/(xi[b]-xi[a])
        z.append((x[i]-((1-t)*x[a]+t*x[b]))/max(r[i],1e-12))
    if not z: return 0.0
    z=np.asarray(z); return float(np.clip(1.4826*np.median(np.abs(z-np.median(z,axis=0))),0,.25))


def _fit(x, r, xi, lam, ops):
    w=_weights(xi)/np.maximum(r*r,1e-24); H=sparse.diags(w,format="csc")
    for op in ops: H += lam/3*(op.T@op)
    if len(x) <= 2: return x.copy()
    free=np.arange(1,len(x)-1); fixed=np.array([0,len(x)-1]); out=x.copy()
    for k in range(3):
        out[free,k]=spsolve(H[free][:,free],w[free]*x[free,k]-H[free][:,fixed]@x[fixed,k])
    return out


def _auto_fit(x,r,xi):
    ops=tuple(_second_difference(xi,r,h) for h in SCALES); target=_noise(x,xi,r)
    if target < 1e-8 or not any(o.shape[0] for o in ops): return 0.0,x.copy()
    lo,hi=0.0,1.0; best=x.copy()
    def run(v):
        y=_fit(x,r,xi,v,ops); d=np.linalg.norm(y-x,axis=1)/np.maximum(r,1e-12)
        return y,float(np.sqrt(np.mean(d*d)))
    best,err=run(hi)
    while err < target and hi < 1e8: hi*=10; best,err=run(hi)
    for _ in range(18):
        mid=hi*.1 if lo==0 else np.sqrt(lo*hi); y,err=run(mid)
        if err < target: lo=mid
        else: hi,best=mid,y
    return hi,best


def _minrad_points(p0,p1,p2):
    a=p1-p0; b=p2-p1; la=np.linalg.norm(a); lb=np.linalg.norm(b)
    if min(la,lb)<1e-15:return 0.0
    angle=np.arccos(np.clip(np.dot(a,b)/(la*lb),-1,1)); q=np.tan(angle/2)
    return float("inf") if q<1e-15 else min(la,lb)/(2*q)


def _minrad(x,i):
    return _minrad_points(x[i-1],x[i],x[i+1])


def _curv_count(x,segs):
    return sum(_minrad(x[s.idx],i)+1e-12 < s.r[i]/KAPPA_R_MAX
               for s in segs for i in range(1,len(s.idx)-1))


def _edges(x,segs):
    out=[]
    for s in segs:
        local=x[s.idx]; arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(local,axis=0),axis=1))]
        for i in range(len(s.idx)-1):
            out.append(_Edge(s.sid,i,int(s.idx[i]),int(s.idx[i+1]),s.r[i],s.r[i+1],arc[i],arc[i+1],arc[-1]))
    return out


def _junction(e,f,u,v,ra,rb,byseg):
    a,b=byseg[e.seg],byseg[f.seg]; common=set((a.n1,a.n2))&set((b.n1,b.n2))
    if not common:return False
    n=next(iter(common)); sa=e.s0+u*(e.s1-e.s0); sb=f.s0+v*(f.s1-f.s0)
    da=sa if a.n1==n else e.length-sa; db=sb if b.n1==n else f.length-sb
    rna=a.r[0] if a.n1==n else a.r[-1]; rnb=b.r[0] if b.n1==n else b.r[-1]
    rn=max(rna,rnb); return da <= rn+ra and db <= rn+rb


def _self_critical(x,e,f,u,v,byseg):
    """Discrete doubly-critical self-distance test for two polyline edges."""
    s=byseg[e.seg]; pa=(1-u)*x[e.a]+u*x[e.b]; pb=(1-v)*x[f.a]+v*x[f.b]
    eps=1e-8; tol=1e-7*max(np.linalg.norm(pb-pa),1e-12)

    def critical(edge,t,point,other):
        chord=point-other
        if eps < t < 1-eps:
            direction=x[edge.b]-x[edge.a]
            return abs(float(np.dot(chord,direction))) <= tol*max(np.linalg.norm(direction),1e-12)
        local=edge.local if t <= eps else edge.local+1
        directions=[]
        if local>0: directions.append(x[s.idx[local-1]]-point)
        if local+1<len(s.idx): directions.append(x[s.idx[local+1]]-point)
        return bool(directions) and all(float(np.dot(chord,d)) >= -tol*max(np.linalg.norm(d),1e-12) for d in directions)

    return critical(e,u,pa,pb) and critical(f,v,pb,pa)


def _candidate_pairs(x,ed,padding):
    """BVH broad phase for all pairs reachable inside the trust region."""
    if not ed:return []
    p=np.asarray([x[e.a] for e in ed]);q=np.asarray([x[e.b] for e in ed])
    pad=np.asarray(padding,float)
    radius_floor=max(np.finfo(float).eps*max(float(np.ptp(x,axis=0).max()),1e-30),np.finfo(float).tiny)
    pad=np.maximum(pad,radius_floor);bvh=CapsuleBVH(p,q,pad,leaf_size=16);pairs=[];stack=[(bvh.root,bvh.root)]
    while stack:
        na,nb=stack.pop();a,b=bvh.nodes[na],bvh.nodes[nb]
        if np.any(a.lo-a.max_radius > b.hi+b.max_radius) or np.any(b.lo-b.max_radius > a.hi+a.max_radius):continue
        la=a.indices is not None;lb=b.indices is not None
        if la and lb:
            if na==nb:
                ids=a.indices
                pairs.extend((int(ids[i]),int(ids[j])) for i in range(len(ids)) for j in range(i+1,len(ids)))
            else:pairs.extend((int(i),int(j)) if i<j else (int(j),int(i)) for i in a.indices for j in b.indices if i!=j)
        elif na==nb:
            stack.extend(((a.left,a.left),(a.left,a.right),(a.right,a.right)))
        elif la:
            stack.extend(((na,b.left),(na,b.right)))
        elif lb:
            stack.extend(((a.left,nb),(a.right,nb)))
        elif np.linalg.norm(a.hi-a.lo)>=np.linalg.norm(b.hi-b.lo):
            stack.extend(((a.left,nb),(a.right,nb)))
        else:stack.extend(((na,b.left),(na,b.right)))
    return np.asarray(list(dict.fromkeys(pairs)),dtype=np.int64).reshape((-1,2))


def _closest_batch(p0,p1,q0,q1):
    """Vectorized equivalent of ``closest_segment_parameters``."""
    u=p1-p0;v=q1-q0;w=p0-q0
    aa=np.einsum("ij,ij->i",u,u);bb=np.einsum("ij,ij->i",u,v);cc=np.einsum("ij,ij->i",v,v)
    dd=np.einsum("ij,ij->i",u,w);ee=np.einsum("ij,ij->i",v,w);den=aa*cc-bb*bb
    tiny=1e-30;s=np.zeros(len(p0));t=np.zeros(len(p0));general=(aa>tiny)&(cc>tiny)
    safe=np.maximum(den,tiny);s[general]=np.clip((bb[general]*ee[general]-cc[general]*dd[general])/safe[general],0,1)
    t[general]=np.clip((aa[general]*ee[general]-bb[general]*dd[general])/safe[general],0,1)
    for _ in range(2):
        s[general]=np.clip((bb[general]*t[general]-dd[general])/aa[general],0,1)
        t[general]=np.clip((bb[general]*s[general]+ee[general])/cc[general],0,1)
    only_q=(aa<=tiny)&(cc>tiny);t[only_q]=np.clip(ee[only_q]/cc[only_q],0,1)
    only_p=(cc<=tiny)&(aa>tiny);s[only_p]=np.clip(-dd[only_p]/aa[only_p],0,1)
    delta=(p0+s[:,None]*u)-(q0+t[:,None]*v)
    return s,t,np.linalg.norm(delta,axis=1)


def _reachable_pairs(x,ed,pairs,cap,tol=0.0):
    """Keep only pairs whose trust regions can possibly make tubes meet."""
    pairs=np.asarray(pairs,dtype=np.int64)
    if not len(pairs):return pairs.reshape((-1,2))
    pairs=np.sort(pairs,axis=1)
    pairs=np.unique(pairs,axis=0)
    p=np.asarray([x[e.a] for e in ed]);q=np.asarray([x[e.b] for e in ed]);mr=np.asarray([max(e.r0,e.r1) for e in ed])
    i,j=pairs[:,0],pairs[:,1];expanded=(1+cap)*mr
    lo=np.minimum(p,q)-expanded[:,None]-tol;hi=np.maximum(p,q)+expanded[:,None]+tol
    keep=np.all(lo[j]<=hi[i],axis=1)&np.all(lo[i]<=hi[j],axis=1);pairs=pairs[keep]
    if not len(pairs):return pairs
    i,j=pairs[:,0],pairs[:,1];_u,_v,d=_closest_batch(p[i],q[i],p[j],q[j])
    return pairs[d <= (1+cap)*(mr[i]+mr[j])+tol]


def _conflicts(x,segs,tol=0.0,candidate_pairs=None):
    ed=_edges(x,segs); byseg={s.sid:s for s in segs}; out=[]
    if not ed:return out,ed
    p=np.asarray([x[e.a] for e in ed]); q=np.asarray([x[e.b] for e in ed]); mr=np.asarray([max(e.r0,e.r1) for e in ed])
    lo=np.minimum(p,q)-mr[:,None]-tol; hi=np.maximum(p,q)+mr[:,None]+tol
    pairs=_candidate_pairs(x,ed,mr+tol) if candidate_pairs is None else candidate_pairs
    pair_array=np.asarray(pairs,dtype=np.int64)
    if pair_array.size:
        pi,pj=pair_array[:,0],pair_array[:,1]
        overlap=np.all(lo[pj]<=hi[pi],axis=1)&np.all(lo[pi]<=hi[pj],axis=1)
        seg_id=np.asarray([e.seg for e in ed]);local_id=np.asarray([e.local for e in ed])
        overlap&=~((seg_id[pi]==seg_id[pj])&(np.abs(local_id[pi]-local_id[pj])<=1))
        pair_array=pair_array[overlap]
    if not len(pair_array):return out,ed
    pi,pj=pair_array[:,0],pair_array[:,1]
    us,vs,distances=_closest_batch(p[pi],q[pi],p[pj],q[pj])
    edge_r0=np.asarray([e.r0 for e in ed]);edge_r1=np.asarray([e.r1 for e in ed])
    ras=edge_r0[pi]+us*(edge_r1[pi]-edge_r0[pi]);rbs=edge_r0[pj]+vs*(edge_r1[pj]-edge_r0[pj])
    clearances=distances-ras-rbs;keep=clearances<tol
    for pair_index in np.where(keep)[0]:
            i,j=int(pi[pair_index]),int(pj[pair_index]);u=float(us[pair_index]);v=float(vs[pair_index]);ra=float(ras[pair_index]);rb=float(rbs[pair_index])
            e,f=ed[i],ed[j]
            if e.seg==f.seg and not _self_critical(x,e,f,u,v,byseg):continue
            if e.seg!=f.seg and _junction(e,f,u,v,ra,rb,byseg):continue
            out.append(_Conflict(i,j,e.seg,f.seg,u,v,float(clearances[pair_index])))
    return out,ed


def _trust(y,raw,r,cap):
    d=y-raw; n=np.linalg.norm(d,axis=1); over=n>cap*r
    if np.any(over): d[over]*=(cap*r[over]/np.maximum(n[over],1e-30))[:,None]; y[:]=raw+d


def _project_curvature(y,segs,fixed):
    for s in segs:
        for i in range(1,len(s.idx)-1):
            gi=int(s.idx[i]); req=s.r[i]/KAPPA_R_MAX
            prev_i=int(s.idx[i-1]);next_i=int(s.idx[i+1])
            if fixed[gi] or _minrad_points(y[prev_i],y[gi],y[next_i])>=req:continue
            before=y[gi].copy(); target=(y[s.idx[i-1]]+y[s.idx[i+1]])/2; lo,hi=0.,1.
            for _ in range(20):
                a=(lo+hi)/2; y[gi]=before+a*(target-before)
                if _minrad_points(y[prev_i],y[gi],y[next_i])>=req:hi=a
                else:lo=a
            y[gi]=before+hi*(target-before)


def _project_clearance(y,segs,fixed,baseline,tol,candidate_pairs):
    cs,ed=_conflicts(y,segs,tol,candidate_pairs)
    active=[c for c in cs if c.key not in baseline]
    for c in active:
        e,f=ed[c.ei],ed[c.ej]; pa=(1-c.u)*y[e.a]+c.u*y[e.b]; pb=(1-c.v)*y[f.a]+c.v*y[f.b]
        d=pa-pb; nd=np.linalg.norm(d)
        if nd<1e-15:
            d=np.cross(y[e.b]-y[e.a],y[f.b]-y[f.a]); nd=np.linalg.norm(d)
            if nd<1e-15:d=np.array([0.,0.,1.]);nd=1
        normal=d/nd; need=-c.clearance+tol
        terms=((e.a,1-c.u),(e.b,c.u),(f.a,-(1-c.v)),(f.b,-c.v)); den=sum(w*w for i,w in terms if not fixed[i])
        if den>1e-20:
            for i,w in terms:
                if not fixed[i]:y[i]+=need*w/den*normal
    return len(active)


def _global_system(raw,segs):
    """Assemble the graph-wide sparse multiscale normal equations."""
    n=len(raw);H=sparse.csc_matrix((n,n));rhs=np.zeros_like(raw);lams=[]
    for s in segs:
        local=raw[s.idx];ops=tuple(_second_difference(s.xi,s.r,h) for h in SCALES)
        lam,_=_auto_fit(local,s.r,s.xi);lams.append(lam)
        w=_weights(s.xi)/np.maximum(s.r*s.r,1e-24)
        local_h=sparse.diags(w,format="csc")
        for op in ops:local_h+=lam/3*(op.T@op)
        coo=local_h.tocoo()
        H+=sparse.coo_matrix((coo.data,(s.idx[coo.row],s.idx[coo.col])),shape=(n,n)).tocsc()
        rhs[s.idx]+=w[:,None]*local
    return H,rhs,lams


def smooth_centerlines_constrained_multiscale(nodes,points,segments):
    """Return smoothed points and a :class:`SmoothingReport`; radii are immutable."""
    del nodes
    pids=list(dict.fromkeys(pid for s in segments for pid in s.get("point_ids",[]) if pid in points)); mp={p:i for i,p in enumerate(pids)}
    if not pids:
        # Keyword form: the positional call here passed 18 arguments to a
        # 17-field dataclass, a TypeError that only an empty graph could reach.
        return dict(points), SmoothingReport(
            converged=True, iterations=0,
            lambda_min=0.0, lambda_median=0.0, lambda_max=0.0,
            max_displacement_radius=0.0, p95_displacement_radius=0.0,
            curvature_violations_before=0, curvature_violations_after=0,
            self_distance_violations_before=0, self_distance_violations_after=0,
            input_overlaps=0, new_branch_conflicts=0, frozen_points=0,
            unresolved_constraints=0, backtrack_fraction=1.0, modified_points=0)
    raw=np.asarray([points[p][:3] for p in pids],float)/1000; radii=np.asarray([points[p][3] for p in pids],float)/1000*config.RADIUS_SCALE
    segs=[]; fixed=np.zeros(len(raw),bool)
    for si,s in enumerate(segments):
        idx=np.asarray([mp[p] for p in s.get("point_ids",[]) if p in mp],int)
        if len(idx)<2:continue
        keep=np.r_[True,np.linalg.norm(np.diff(raw[idx],axis=0),axis=1)>1e-12];idx=idx[keep]
        if len(idx)<2:continue
        rr=radii[idx];segs.append(_Seg(si,int(s["node1"]),int(s["node2"]),idx,rr,_xi(raw[idx],rr)));fixed[idx[[0,-1]]]=True
    cap=float(getattr(config,"CENTERLINE_MAX_DRIFT_RADIUS_FACTOR",.25))
    # NOTE on a fix that was tried and rejected. These clearance tests compare
    # CAPSULE TUBES, while what gets meshed is the smooth-min BLEND of those
    # capsules, which sits outside the hard union by up to BLEND_BULGE_CAP_MM.
    # Inflating the tolerance by 2*that bound to close the gap was measured and
    # made the surface WORSE (drift 0.10: left tree 0 -> 10 self-intersecting
    # pairs, right tree 157 -> 178) while changing the smoothing by ~1%.
    #
    # The gap is not tolerance-sized. _junction() exempts any conflict between
    # segments sharing a node, so no clearance test runs near a bifurcation at
    # all -- which is exactly where the blend is most complex and where moving a
    # daughter folds it. Closing this properly needs the blended field itself
    # evaluated near junctions, not a wider tube test. Until then the only
    # trustworthy signal is the meshed self-intersection count, so the drift cap
    # is chosen empirically against mesh_validation.
    raw_edges=_edges(raw,segs);possible_pairs=_candidate_pairs(raw,raw_edges,[(1+cap)*max(e.r0,e.r1) for e in raw_edges])
    possible_pairs=_reachable_pairs(raw,raw_edges,possible_pairs,cap)
    before_curv=_curv_count(raw,segs); base,ed=_conflicts(raw,segs,candidate_pairs=possible_pairs); basekeys={c.key for c in base}; self_before=sum(c.si==c.sj for c in base)
    constraint_pairs=np.asarray([pair for pair in possible_pairs if tuple(pair) not in basekeys],dtype=np.int64).reshape((-1,2))
    # Preserve-only policy: freeze capsule endpoints and a one-radius (Delta xi <= 1) halo.
    for c in base:
        for ei in (c.ei,c.ej):
            e=ed[ei];s=next(z for z in segs if z.sid==e.seg)
            for seed in (e.local,e.local+1):fixed[s.idx[np.abs(s.xi-s.xi[seed])<=1+1e-12]]=True
    H,rhs,lams=_global_system(raw,segs)
    diagonal=H.diagonal();positive=diagonal[diagonal>0]
    # rho carries the same inverse-length-squared units as H.  An absolute
    # floor here would break global scale equivariance.
    rho=float(np.median(positive)) if len(positive) else 1.0
    solve=factorized((H+rho*sparse.eye(len(raw),format="csc")).tocsc())
    x=raw.copy();y=raw.copy();dual=np.zeros_like(raw)
    scale=max(np.ptp(raw,axis=0).max(),radii.max(),1e-12);tol=256*np.finfo(float).eps*scale
    converged=False; iterations=0
    for outer in range(20):
        old=y.copy()
        for axis in range(3):x[:,axis]=solve(rhs[:,axis]+rho*(y[:,axis]-dual[:,axis]))
        trial=x+dual
        # One projection of each nonlinear set per sequential-convex outer
        # iteration. Repeating a stale linearization here is both expensive and
        # less stable than re-solving/relinearizing on the next outer step.
        _trust(trial,raw,radii,cap);trial[fixed]=raw[fixed]
        _project_curvature(trial,segs,fixed)
        _trust(trial,raw,radii,cap);trial[fixed]=raw[fixed]
        _project_clearance(trial,segs,fixed,basekeys,tol,constraint_pairs)
        _trust(trial,raw,radii,cap);trial[fixed]=raw[fixed]
        y=trial;dual+=x-y
        iterations=outer+1
        update=np.max(np.linalg.norm(y-old,axis=1)/np.maximum(radii,1e-12))
        primal=np.max(np.linalg.norm(x-y,axis=1)/np.maximum(radii,1e-12))
        if max(update,primal)<1e-4:converged=True;break
    # Certified line search, backtracked PER SEGMENT.
    #
    # The certificate is the same as before: admit no NEW inter-branch or
    # nonlocal self-contact anywhere. What changes is its granularity. A single
    # global fraction is dictated by the worst location in the whole graph, so
    # on a tortuous tree one unavoidable contact vetoes the correction
    # everywhere. Measured on the LADAF left+right graph: 18,858 pre-existing
    # tube overlaps drove the accepted global fraction to 0.018, which applied
    # 0.4% of the available drift and fixed 4 of 18,368 curvature violations --
    # the optimiser was computing a good centreline and then discarding it.
    #
    # Backtracking each segment independently keeps the guarantee (a conflict
    # still forces both implicated segments back) while letting the conflict-free
    # majority take their full step. Interior points belong to exactly one
    # segment -- shared junction points are pinned by ``fixed`` -- and the
    # min-reduction below is a belt-and-braces guard for any point that is
    # nonetheless claimed twice.
    alpha = {s.sid: 1.0 for s in segs}

    def _apply(alpha_map: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        a_pt = np.ones(len(raw))
        for s in segs:
            np.minimum.at(a_pt, s.idx, alpha_map[s.sid])
        z = raw + a_pt[:, None] * (y - raw)
        z[fixed] = raw[fixed]
        return z, a_pt

    final, alpha_pt = _apply(alpha)
    for _ in range(_BACKTRACK_ROUNDS):
        cs, _ed = _conflicts(final, segs, tol, constraint_pairs)
        if not cs:
            break
        # ``constraint_pairs`` already excludes the baseline pairs, so every
        # conflict here is new by construction.
        moved_back = False
        for sid in {c.si for c in cs} | {c.sj for c in cs}:
            a = alpha.get(sid, 0.0)
            if a > 0.0:
                # Halve, snapping to zero once the step is too small to matter,
                # so a hopeless segment reverts outright instead of consuming
                # rounds. Geometric decay bounds the loop.
                alpha[sid] = a * 0.5 if a * 0.5 >= _ALPHA_FLOOR else 0.0
                moved_back = True
        if not moved_back:
            break
        final, alpha_pt = _apply(alpha)
    free = ~fixed
    frac = float(alpha_pt[free].mean()) if free.any() else 1.0
    final[fixed]=raw[fixed];cs,_=_conflicts(final,segs,candidate_pairs=possible_pairs);self_after=sum(c.si==c.sj for c in cs);nnew=sum(c.key not in basekeys and c.si!=c.sj for c in cs);after_curv=_curv_count(final,segs)
    disp=np.linalg.norm(final-raw,axis=1)/np.maximum(radii,1e-12);out=dict(points);modified=0
    for p,i in mp.items():
        old=points[p];xyz=final[i]*1000;modified+=bool(np.linalg.norm(xyz-np.asarray(old[:3]))>1e-9);out[p]=(float(xyz[0]),float(xyz[1]),float(xyz[2]),*old[3:])
    new_self=sum(c.key not in basekeys and c.si==c.sj for c in cs)
    lv=np.asarray(lams or [0.]); unresolved=after_curv+new_self+nnew
    return out,SmoothingReport(converged,iterations,float(lv.min()),float(np.median(lv)),float(lv.max()),float(disp.max(initial=0)),float(np.percentile(disp,95)),before_curv,after_curv,self_before,self_after,len(base),nnew,int(fixed.sum()),unresolved,frac,int(modified))


__all__=["SmoothingReport","smooth_centerlines_constrained_multiscale"]
