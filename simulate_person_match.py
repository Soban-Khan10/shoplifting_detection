import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2


DEFAULT_OUTPUT_DIR = Path("outputs/person_match_demo")
DEFAULT_DATABASE = Path("data/fake_people_database.json")
FAKE_PROFILES = [
    {
        "person_id": "P001",
        "name": "Alex Mercer",
        "age": 34,
        "gender": "male",
        "last_seen_store": "North Plaza Market",
        "previous_incidents": 2,
        "risk_level": "medium",
        "notes": "Demo profile for dashboard testing only.",
    },
    {
        "person_id": "P002",
        "name": "Maya Stone",
        "age": 29,
        "gender": "female",
        "last_seen_store": "Riverside Outlet",
        "previous_incidents": 0,
        "risk_level": "low",
        "notes": "Simulated profile with no prior incidents.",
    },
    {
        "person_id": "P003",
        "name": "Jordan Hale",
        "age": 41,
        "gender": "non-binary",
        "last_seen_store": "City Center Grocery",
        "previous_incidents": 1,
        "risk_level": "medium",
        "notes": "Synthetic identity used for mock matching.",
    },
    {
        "person_id": "P004",
        "name": "Riley Quinn",
        "age": 23,
        "gender": "female",
        "last_seen_store": "Harbor Mall",
        "previous_incidents": 3,
        "risk_level": "high",
        "notes": "High-risk demo profile for UI validation.",
    },
    {
        "person_id": "P005",
        "name": "Noah Bennett",
        "age": 38,
        "gender": "male",
        "last_seen_store": "Greenfield Supermarket",
        "previous_incidents": 0,
        "risk_level": "low",
        "notes": "Placeholder only, not a real person.",
    },
    {
        "person_id": "P006",
        "name": "Casey Brooks",
        "age": 46,
        "gender": "female",
        "last_seen_store": "Downtown Corner Store",
        "previous_incidents": 4,
        "risk_level": "high",
        "notes": "Used for simulated incident workflows.",
    },
    {
        "person_id": "P007",
        "name": "Taylor Reed",
        "age": 31,
        "gender": "male",
        "last_seen_store": "Airport Convenience",
        "previous_incidents": 1,
        "risk_level": "medium",
        "notes": "Fictional record for testing only.",
    },
    {
        "person_id": "P008",
        "name": "Avery Cole",
        "age": 27,
        "gender": "female",
        "last_seen_store": "Market Square Mini Mart",
        "previous_incidents": 0,
        "risk_level": "low",
        "notes": "No real-world identity represented.",
    },
    {
        "person_id": "P009",
        "name": "Drew Carter",
        "age": 52,
        "gender": "male",
        "last_seen_store": "Sunset Retail Hub",
        "previous_incidents": 2,
        "risk_level": "medium",
        "notes": "Synthetic demo-only profile.",
    },
    {
        "person_id": "P010",
        "name": "Sydney Park",
        "age": 36,
        "gender": "female",
        "last_seen_store": "Oak Street Pharmacy",
        "previous_incidents": 5,
        "risk_level": "high",
        "notes": "Fictional profile for UI and reporting tests.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Simulated person matching demo for a suspicious key frame.")
    parser.add_argument("--image", required=True, help="Path to suspicious key frame image.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for simulated outputs.")
    parser.add_argument("--database", default=str(DEFAULT_DATABASE), help="Path to fake people database JSON.")
    return parser.parse_args()


def ensure_database(database_path):
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if not database_path.exists():
        with database_path.open("w") as f:
            json.dump(FAKE_PROFILES, f, indent=2)
    with database_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Fake people database is invalid: {database_path}")
    return data


def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read suspicious key frame image: {image_path}")
    return image


def crop_center_region(image):
    height, width = image.shape[:2]
    crop_width = max(1, int(width * 0.5))
    crop_height = max(1, int(height * 0.5))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    return image[top:bottom, left:right]


def select_profile(image_path, profiles):
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(profiles)
    return profiles[index]


def simulated_confidence(image_path):
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
    seed = int(digest[8:16], 16)
    rng = random.Random(seed)
    return round(rng.uniform(0.75, 0.95), 3)


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser()
    output_dir = Path(args.output_dir)
    database_path = Path(args.database).expanduser()

    if not image_path.exists():
        raise FileNotFoundError(f"Suspicious key frame image not found: {image_path}")

    profiles = ensure_database(database_path)
    image = load_image(image_path)
    crop = crop_center_region(image)

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / "suspect_crop.jpg"
    result_json_path = output_dir / "simulated_match_result.json"

    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Could not write suspect crop image: {crop_path}")

    profile = select_profile(image_path, profiles)
    confidence = simulated_confidence(image_path)
    result = {
        "simulated_only": True,
        "warning": "Simulated demo only, not real face recognition.",
        "input_image": str(image_path),
        "suspect_crop_path": str(crop_path),
        "database_path": str(database_path),
        "matched_profile": profile,
        "simulated_confidence": confidence,
    }
    with result_json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"input_image_path: {image_path}")
    print(f"suspect_crop_path: {crop_path}")
    print(f"matched_fake_person_id: {profile['person_id']}")
    print(f"matched_fake_name: {profile['name']}")
    print(f"simulated_confidence: {confidence:.3f}")
    print(f"result_json_path: {result_json_path}")
    print("warning: simulated demo only, not real face recognition")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
