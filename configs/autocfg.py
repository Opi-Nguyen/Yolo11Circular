
import os
import yaml
import argparse

class YOLOConfig:
    """
    Load default config, then override with optional YAML, then override with CLI args (highest priority).
    Stores a flat dict of keys -> values for simplicity.
    """
    def __init__(self, base_config_path: str, override_config_path: str = None):
        with open(base_config_path, "r") as f:
            base_cfg = yaml.safe_load(f) or {}
        self.cfg = {k: self._convert_value(v) for k, v in base_cfg.items()}

        if override_config_path:
            if not os.path.isfile(override_config_path):
                raise FileNotFoundError(f"Override config not found: {override_config_path}")
            print(f"📌 Overriding config with: {override_config_path}")
            with open(override_config_path, "r") as f:
                override_cfg = yaml.safe_load(f) or {}
            self.cfg.update({k: self._convert_value(v) for k, v in override_cfg.items()})

    def _convert_value(self, value):
        # Convert strings that look like ints/floats; leave others intact
        if isinstance(value, str):
            try:
                if value.lower() in ("true", "false"):
                    return value.lower() == "true"
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
        return value

    def get(self, key: str, default=None):
        return self.cfg.get(key, default)

    def set(self, key: str, value):
        self.cfg[key] = value

    def update_from_args(self, args: argparse.Namespace):
        for key, value in vars(args).items():
            if value is not None:
                self.cfg[key.replace('-', '_')] = value  # normalize dashes to underscores

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False, sort_keys=False)


def load_config(default_path: str, override_path: str = None, args: argparse.Namespace = None) -> YOLOConfig:
    cfg = YOLOConfig(default_path, override_path)
    if args is not None:
        cfg.update_from_args(args)
    return cfg


def create_unique_folder(base_folder_path: str) -> str:
    # If base path doesn't exist, create and return
    if not os.path.exists(base_folder_path):
        os.makedirs(base_folder_path, exist_ok=True)
        print(f"Created folder: {base_folder_path}")
        return base_folder_path

    # Otherwise, add suffix _1, _2, ...
    count = 1
    while True:
        new_folder_path = f"{base_folder_path}_{count}"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"Created folder: {new_folder_path}")
            return new_folder_path
        count += 1


def _in_run_dir(run_dir: str, maybe_path: str, default_name: str) -> str:
    """
    Nếu maybe_path là None hoặc rỗng -> dùng default_name.
    Nếu maybe_path có kèm thư mục -> chỉ lấy basename.
    Trả về: run_dir / <basename>
    """
    name = os.path.basename(maybe_path) if maybe_path else default_name
    return os.path.join(run_dir, name)