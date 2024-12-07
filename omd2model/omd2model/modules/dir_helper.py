
from omd2model.modules import omd, bmd, smd


def select_matrix_dir(dir_type, K, A = None, band = None,event_dim=None,site_name="trans_ka",prior={},device="cpu", ):
    if dir_type.lower() == "smd":
        assert K != None and A != None, "need to provide K and A"
        matrix_ka = smd.SMD(K, A, prior, site_name=site_name, event_dim=event_dim, device=device)
    elif dir_type.lower() == "bmd":
        assert K != None and band != None, "need to provide K and band"
        matrix_ka = bmd.BMD(K, band, prior, site_name=site_name, event_dim=event_dim, device=device)
    elif dir_type.lower() == "omd":
        assert K != None and A != None, "need to provide K and A"
        matrix_ka = omd.OMD(K, A, prior, site_name=site_name, event_dim=event_dim, device=device)
    else:
        print(f"unknown dir_type {dir_type}")
    return matrix_ka