import shortfin as sf


def get_selected_devices(sb: sf.SystemBuilder, device_ids=None):
    available = sb.available_devices
    selected = []
    if device_ids is not None:
        for d in device_ids:
            if not isinstance(d, int):
                continue
            if d >= len(available) or d < 0:
                raise ValueError(
                    f"Requested more device ids ({device_ids}) than available ({available})."
                )
        for did in device_ids:
            if isinstance(did, str):
                try:
                    did = int(did)
                except ValueError:
                    did = did
            if did in available:
                selected.append(did)
            elif isinstance(did, int):
                selected.append(available[did])
            else:
                raise ValueError(f"Device id {did} could not be parsed.")
    else:
        selected = available
    return selected
