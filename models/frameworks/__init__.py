def get_model(args, framework):
    if framework == "NeuMesh":
        from .neumesh import get_neumesh_model as get_model
    else:
        raise NotImplementedError
    return get_model(args)
