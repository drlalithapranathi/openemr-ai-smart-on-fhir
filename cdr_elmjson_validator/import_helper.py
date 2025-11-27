#!/usr/bin/env python3
"""
ELM Validation Import Helper Script

Validates all ELM files in CQL Services using Modal/Llama.
Tracks validation status and returns valid libraries for OpenEMR import.

Usage:
    python3 import_helper.py                    # Validate all unvalidated libraries
    python3 import_helper.py --all              # Re-validate all libraries
    python3 import_helper.py <library_name>     # Validate specific library
"""

import os
import sys
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime

# Configuration
CQL_SERVICES_PATH = os.environ.get(
    "CQL_SERVICES_PATH",
    str(Path.home() / "AHRQ-CDS-Connect-CQL-SERVICES")
)

MODAL_VALIDATOR_URL = "https://drlalithapranathi--elm-validator-validate.modal.run"


def get_file_hash(file_path):
    """Calculate MD5 hash of a file to detect changes."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_validation_status(library_dir, elm_file):
    """
    Check if library has been validated and if ELM file has changed.

    Returns:
        dict: {"validated": bool, "result": dict or None, "hash": str}
    """
    validation_file = library_dir / ".validation.json"
    current_hash = get_file_hash(elm_file)

    if not validation_file.exists():
        return {"validated": False, "result": None, "hash": current_hash}

    try:
        with open(validation_file, 'r') as f:
            validation_data = json.load(f)

        # Check if file has changed since last validation
        if validation_data.get("elm_hash") != current_hash:
            return {"validated": False, "result": None, "hash": current_hash}

        return {
            "validated": True,
            "result": validation_data.get("result"),
            "hash": current_hash
        }
    except Exception:
        return {"validated": False, "result": None, "hash": current_hash}


def save_validation_status(library_dir, elm_file, validation_result):
    """Save validation result and ELM hash to track validation status."""
    validation_file = library_dir / ".validation.json"

    validation_data = {
        "elm_hash": get_file_hash(elm_file),
        "validated_at": datetime.now().isoformat(),
        "result": validation_result
    }

    with open(validation_file, 'w') as f:
        json.dump(validation_data, f, indent=2)


def get_all_libraries():
    """
    Find all library directories in CQL Services.

    Returns:
        list: List of library directory names
    """
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"

    if not libraries_path.exists():
        raise FileNotFoundError(f"Libraries path not found: {libraries_path}")

    # Get all subdirectories
    libraries = []
    for item in libraries_path.iterdir():
        if item.is_dir():
            # Skip helper libraries
            if item.name not in ["FHIRHelpers", "Commons", "Conversions"]:
                libraries.append(item.name)

    return libraries


def find_main_elm_file(library_dir):
    """
    Find the main ELM JSON file in a library directory.
    Skips helper libraries like FHIRHelpers, Commons, Conversions.

    Args:
        library_dir: Path to the library directory

    Returns:
        Path to the main ELM file, or None if not found
    """
    for json_file in library_dir.glob("*.json"):
        # Skip helper libraries
        if any(skip in json_file.name for skip in ["FHIRHelpers", "Commons", "Conversions", "Shared"]):
            continue
        # Found the main library file
        return json_file
    return None


def get_elm_from_local(library_name):
    """
    Fetch ELM JSON from local CQL Services file system.

    Args:
        library_name: Name of the library directory (e.g., "statin-use")

    Returns:
        ELM JSON dict
    """
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"
    library_dir = libraries_path / library_name

    if not library_dir.exists():
        raise FileNotFoundError(f"Library directory not found: {library_dir}")

    # Find the main ELM file
    elm_file = find_main_elm_file(library_dir)
    if not elm_file:
        raise FileNotFoundError(f"No main ELM file found in {library_dir}")

    print(f"  Found ELM file: {elm_file.name}")

    with open(elm_file, 'r') as f:
        return json.load(f)


def send_to_modal(elm_json, library_name):
    """
    Send ELM JSON to Modal validator via HTTP.

    Args:
        elm_json: The ELM JSON dictionary
        library_name: Name of the library

    Returns:
        Validation result dict
    """
    payload = {
        "elm_json": elm_json,
        "library_name": library_name
    }

    response = requests.post(
        MODAL_VALIDATOR_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=180  # 3 minutes (Llama can be slow on first run)
    )
    response.raise_for_status()

    return response.json()


def validate_library(library_name, force=False):
    """
    Complete validation pipeline for a single library.

    Args:
        library_name: Name of the library to validate
        force: If True, re-validate even if already validated

    Returns:
        Validation result dict with library name
    """
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"
    library_dir = libraries_path / library_name

    if not library_dir.exists():
        return {
            "library": library_name,
            "valid": False,
            "errors": [f"Library directory not found: {library_dir}"],
            "warnings": [],
            "source": "error"
        }

    # Find main ELM file
    elm_file = find_main_elm_file(library_dir)
    if not elm_file:
        return {
            "library": library_name,
            "valid": False,
            "errors": [f"No main ELM file found in {library_dir}"],
            "warnings": [],
            "source": "error"
        }

    # Check if already validated (unless force)
    if not force:
        status = get_validation_status(library_dir, elm_file)
        if status["validated"]:
            print(f"  [OK] {library_name}: Already validated (cached)")
            result = status["result"]
            result["library"] = library_name
            return result

    # Load and validate
    print(f"  --> {library_name}: Validating...")

    try:
        with open(elm_file, 'r') as f:
            elm_json = json.load(f)

        result = send_to_modal(elm_json, library_name)
        result["library"] = library_name

        # Save validation status
        save_validation_status(library_dir, elm_file, result)

        status_icon = "[OK]" if result.get("valid") else "[FAIL]"
        print(f"  {status_icon} {library_name}: {'Valid' if result.get('valid') else 'Invalid'}")

        return result

    except Exception as e:
        print(f"  [FAIL] {library_name}: Error - {e}")
        return {
            "library": library_name,
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "source": "error"
        }


def validate_all_libraries(force=False):
    """
    Validate all libraries in CQL Services.

    Args:
        force: If True, re-validate all libraries

    Returns:
        dict: {"valid_libraries": [...], "invalid_libraries": [...]}
    """
    print("=" * 70)
    print("ELM Validation - All Libraries")
    print("=" * 70)
    print(f"CQL Services Path: {CQL_SERVICES_PATH}")
    print(f"Mode: {'Re-validate all' if force else 'Validate new/updated only'}")
    print()

    libraries = get_all_libraries()
    print(f"Found {len(libraries)} libraries\n")

    valid_libraries = []
    invalid_libraries = []

    for library_name in libraries:
        result = validate_library(library_name, force=force)

        if result.get("valid"):
            valid_libraries.append({
                "name": library_name,
                "source": result.get("source"),
                "warnings": result.get("warnings", [])
            })
        else:
            invalid_libraries.append({
                "name": library_name,
                "errors": result.get("errors", []),
                "source": result.get("source")
            })

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[OK] Valid libraries: {len(valid_libraries)}")
    if valid_libraries:
        for lib in valid_libraries:
            print(f"   - {lib['name']}")

    print(f"\n[FAIL] Invalid libraries: {len(invalid_libraries)}")
    if invalid_libraries:
        for lib in invalid_libraries:
            print(f"   - {lib['name']} ({len(lib['errors'])} errors)")

    print("=" * 70)

    return {
        "valid_libraries": valid_libraries,
        "invalid_libraries": invalid_libraries,
        "total": len(libraries),
        "validated_at": datetime.now().isoformat()
    }


def main():
    """CLI entrypoint."""
    # Parse arguments
    force = False
    json_output = False
    library_name = None

    for arg in sys.argv[1:]:
        if arg == "--all":
            force = True
        elif arg == "--json":
            json_output = True
        elif not arg.startswith("--"):
            library_name = arg

    try:
        # Mode 1: Validate specific library
        if library_name:
            result = validate_library(library_name, force=force)

            if json_output:
                print(json.dumps(result, indent=2))

            sys.exit(0 if result.get('valid') else 1)

        # Mode 2: Validate all libraries
        else:
            result = validate_all_libraries(force=force)

            if json_output:
                print(json.dumps(result, indent=2))

            # Exit 0 if there are any valid libraries, 1 if all failed
            sys.exit(0 if len(result['valid_libraries']) > 0 else 1)

    except FileNotFoundError as e:
        error = {"error": str(e)}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nERROR: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        error = {"error": f"Modal request failed: {e}"}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nERROR: Modal request failed: {e}")
        sys.exit(1)
    except Exception as e:
        error = {"error": str(e)}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nERROR: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()