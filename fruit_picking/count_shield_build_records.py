"""Apply fixed decoration offsets to builder records before component attachment."""
from contextlib import contextmanager

import numpy as np


def record_signature(records):
    return [{'type':r.type,'pose':np.r_[r.pose.p,r.pose.q].tolist(),
             'scale':np.asarray(r.scale).tolist()} for r in records]


@contextmanager
def prebuild_basket_offsets(scene, shifts):
    """Intercept only this scene's basket build, then restore the factory exactly."""
    import sapien
    original_factory=scene.create_actor_builder
    owned='create_actor_builder' in vars(scene)
    original_attribute=vars(scene).get('create_actor_builder')
    audit={'basket_builds':0,'applied_shifts':[]}

    def factory(*args,**kwargs):
        builder=original_factory(*args,**kwargs)
        original_build=builder.build_static
        def build_static(name):
            if name=='basket':
                assert audit['basket_builds']==0,'Apply the basket decoration offset once'
                assert len(builder.visual_records)==len(shifts)==5
                assert all(r.type=='box' for r in builder.visual_records)
                before=record_signature(builder.collision_records)
                audit['visual_before']=record_signature(builder.visual_records)
                for record,shift in zip(builder.visual_records,shifts):
                    pose=record.pose
                    # This is a Python VisualShapeRecord, not an attached RenderShape.
                    record.pose=sapien.Pose(p=np.asarray(pose.p)+shift,q=pose.q)
                audit['visual_after']=record_signature(builder.visual_records)
                after=record_signature(builder.collision_records)
                assert before==after,'Visual-record offsets must leave collision records intact'
                audit.update(basket_builds=1,applied_shifts=shifts,
                             collision_before=before,collision_after=after)
            return original_build(name)
        builder.build_static=build_static
        return builder

    scene.create_actor_builder=factory
    try:yield audit
    finally:
        if owned:scene.create_actor_builder=original_attribute
        else:del scene.create_actor_builder
