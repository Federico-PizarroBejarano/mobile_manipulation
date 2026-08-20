"""Utilities for parsing general configuration dictionaries."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rospkg
import xacro
import yaml


def recursive_dict_update(default, custom):
    """Recursively merge custom dictionary into default dictionary.

    Args:
        default (dict): Default dictionary (will be modified in-place).
        custom (dict): Custom dictionary with overrides.

    Returns:
        dict: Merged dictionary (same object as default).
    """
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError("Params of recursive_update should be dicts")

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(default.get(key), dict):
            default[key] = recursive_dict_update(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default


def load_config(path, depth=0, max_depth=5):
    """Load configuration file with support for included files.

    `depth` and `max_depth` arguments are provided to protect against
    unexpectedly deep or infinite recursion through included files.

    Args:
        path (str): Path to configuration file.
        depth (int): Current recursion depth (used internally).
        max_depth (int): Maximum allowed recursion depth.

    Returns:
        dict: Merged configuration dictionary.

    Raises:
        Exception: If maximum inclusion depth is exceeded.
    """
    if depth > max_depth:
        raise Exception(f"Maximum inclusion depth {max_depth} exceeded.")

    with open(parse_path(path)) as f:
        d = yaml.safe_load(f)

    # get the includes while also removing them from the dict
    includes = d.pop("include", [])

    # construct a dict of everything included
    includes_dict = {}
    for include in includes:
        path = parse_ros_path(include)
        include_dict = load_config(path, depth=depth + 1)

        # nest the include under `key` if specified
        if "key" in include:
            include_dict = {include["key"]: include_dict}

        # update the includes dict and reassign
        includes_dict = recursive_dict_update(includes_dict, include_dict)

    # now add in the info from this file
    d = recursive_dict_update(includes_dict, d)
    return d


def parse_number(x, dtype=float):
    """Parse a number from the config.

    If the number can be converted to a float, then it is and is returned.
    Otherwise, check if it ends with "pi" and convert it to a float that is a
    multiple of pi.

    Args:
        x (str or number): Number or string to parse (e.g., "1.5", "2pi").
        dtype (type): Target data type (default: float).

    Returns:
        float or dtype: Parsed number.

    Raises:
        ValueError: If the string cannot be parsed as a number.
    """
    try:
        return dtype(x)
    except ValueError:
        if isinstance(x, str) and x.endswith("pi"):
            return float(x[:-2]) * np.pi
        else:
            raise ValueError(f"Could not parse {x} as a number.")


def parse_array_element(x):
    """Parse a single array element (float, pi-scaled, or repeated value).

    Args:
        x (str): String to parse. Can be:
            - A float (e.g., "1.5")
            - Pi-scaled (e.g., "2pi" for 2*pi)
            - Repeated (e.g., "1.0rep5" for [1.0, 1.0, 1.0, 1.0, 1.0])

    Returns:
        list: Parsed value as a list.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        return [float(x)]
    except ValueError:
        if x.endswith("pi"):
            return [float(x[:-2]) * np.pi]
        if "rep" in x:
            y, n = x.split("rep")
            return float(y) * np.ones(int(n))
        raise ValueError(f"Could not convert {x} to array element.")


def parse_array(a):
    """Parse a one-dimensional iterable into a numpy array.

    Args:
        a (iterable): One-dimensional iterable of array elements (can include floats, "pi"-scaled values, or repeated values).

    Returns:
        ndarray: Parsed numpy array.
    """
    subarrays = []
    for x in a:
        subarrays.append(parse_array_element(x))
    return np.concatenate(subarrays)


def parse_path(path):
    """Parse and resolve file path, handling ROS package and environment variable references.

    Args:
        path (str): Path string that may contain:
            - $(rospack find <package>) commands
            - Environment variables (e.g., $HOME)

    Returns:
        str or None: Resolved path string, or None if rospack command fails.
    """
    # Regular expression to match the $(rospack find <package>) command
    rospack_pattern = r"\$\((rospack find \w+)\)"

    # Search for the $(rospack find <package>) command
    match = re.search(rospack_pattern, path)
    if match:
        rospack_command = match.group(
            1
        )  # Extract the rospack command (e.g., "rospack find mm_control")

        try:
            # Use subprocess to run the rospack command (e.g., rospack find mm_control)
            package_path = subprocess.check_output(
                rospack_command.split(), text=True
            ).strip()

            # Replace the $(rospack find <package>) part with the actual path
            resolved_path = re.sub(rospack_pattern, package_path, path)

            # Expand any environment variables in the resolved path
            expanded_path = os.path.expandvars(resolved_path)

            return expanded_path
        except subprocess.CalledProcessError as e:
            print(f"Error running command {rospack_command}: {e}")
            return None
    else:
        return os.path.expandvars(path)


def parse_ros_path(d, as_string=True):
    """Resolve full path from a dict containing ROS package and relative path.

    Args:
        d (dict): Dictionary with keys "package" (str) and "path" (str).
        as_string (bool): If True, return path as string; if False, return Path object.

    Returns:
        str or Path: Resolved full path.
    """
    package_name = d["package"]
    relative_path = d["path"]

    # Try ROS first (if available)
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        path = Path(package_path) / relative_path
    except (rospkg.ResourceNotFound, AttributeError):
        # ROS not available - try to find package relative to current directory
        # This works when running from repository root (e.g., /home/federico/mobile_manipulation)
        current_dir = Path.cwd()
        package_dir = current_dir / package_name

        if package_dir.exists() and package_dir.is_dir():
            path = package_dir / relative_path
        else:
            raise ValueError(
                f"Could not find ROS package '{package_name}' and package directory "
                f"not found at '{package_dir}'. Make sure you're running from the "
                f"repository root or install ROS."
            )

    if as_string:
        path = path.as_posix()
    return path


def xacro_include(path):
    """Generate xacro include directive string.

    Args:
        path (str): Path to file to include.

    Returns:
        str: Xacro include directive XML string.
    """
    return f"""
    <xacro:include filename="{path}" />
    """


def preprocess_find_commands(text):
    """Replace all $(find package_name)/path commands in text with resolved paths.

    Also handles <xacro:include filename="$(find ...)"/> directives and all other
    occurrences of $(find ...) in attributes, macro arguments, etc.

    Args:
        text (str): Text that may contain $(find ...) commands.

    Returns:
        str: Text with all $(find ...) commands resolved.
    """
    # More aggressive pattern that matches $(find package)/path in any context
    # This pattern is designed to catch $(find ...) in:
    # - Attributes: filename="$(find pkg)/path"
    # - Macro arguments: $(find pkg)/path
    # - Text content: <value>$(find pkg)/path</value>
    # The pattern uses a non-greedy match to stop at quotes, spaces, or closing parens
    find_pattern = r"\$\(find\s+(\w+)\)([^)\s\"'<>]*)"

    def replace_all_find(match):
        package_name = match.group(1)
        relative_path = match.group(2).lstrip("/")

        try:
            resolved_path = _resolve_package_path(package_name, relative_path)
            return resolved_path
        except ValueError:
            # If we couldn't resolve it, leave it as-is (will cause error later)
            return match.group(0)

    # Replace all $(find ...) patterns in the text
    # Use a while loop to handle nested or multiple occurrences
    max_iterations = 10
    iteration = 0
    while iteration < max_iterations:
        new_text = re.sub(find_pattern, replace_all_find, text)
        if new_text == text:
            # No more replacements needed
            break
        text = new_text
        iteration += 1

    return text


def _resolve_package_path(package_name, relative_path):
    """Helper function to resolve a package name and relative path to an absolute path.

    Args:
        package_name (str): ROS package name.
        relative_path (str): Relative path within the package.

    Returns:
        str: Resolved absolute path.
    """
    # Try ROS first
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        return str(Path(package_path) / relative_path)
    except (rospkg.ResourceNotFound, AttributeError):
        # Fall back to searching relative paths
        current_dir = Path.cwd()

        # Try current directory first
        package_dir = current_dir / package_name
        if package_dir.exists() and package_dir.is_dir():
            # Check if this is actually a ROS package (has package.xml) or a repo with nested package
            package_xml = package_dir / "package.xml"
            if package_xml.exists():
                # It's a ROS package
                return str(package_dir / relative_path)
            else:
                # Might be a repo with nested package structure (e.g., ur_description/ur_description/)
                nested_package_dir = package_dir / package_name
                nested_package_xml = nested_package_dir / "package.xml"
                if nested_package_xml.exists():
                    return str(nested_package_dir / relative_path)
                # Fall back to treating it as a package anyway
                return str(package_dir / relative_path)

        # Try parent directory (common for sibling packages in workspace)
        parent_dir = current_dir.parent
        package_dir = parent_dir / package_name
        if package_dir.exists() and package_dir.is_dir():
            # Check if this is actually a ROS package or a repo with nested package
            package_xml = package_dir / "package.xml"
            if package_xml.exists():
                return str(package_dir / relative_path)
            else:
                # Might be a repo with nested package structure
                nested_package_dir = package_dir / package_name
                nested_package_xml = nested_package_dir / "package.xml"
                if nested_package_xml.exists():
                    return str(nested_package_dir / relative_path)
                # Fall back to treating it as a package anyway
                return str(package_dir / relative_path)

        # Try common workspace locations
        for workspace_pattern in ["src", "catkin_ws/src", "workspace/src"]:
            workspace_dir = current_dir.parent / workspace_pattern / package_name
            if workspace_dir.exists() and workspace_dir.is_dir():
                # Check if this is actually a ROS package or a repo with nested package
                package_xml = workspace_dir / "package.xml"
                if package_xml.exists():
                    return str(workspace_dir / relative_path)
                else:
                    # Might be a repo with nested package structure
                    nested_package_dir = workspace_dir / package_name
                    nested_package_xml = nested_package_dir / "package.xml"
                    if nested_package_xml.exists():
                        return str(nested_package_dir / relative_path)
                    # Fall back to treating it as a package anyway
                    return str(workspace_dir / relative_path)

        # Try common ROS installation locations
        # Check ROS_PACKAGE_PATH environment variable
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
        if ros_package_path:
            for path in ros_package_path.split(":"):
                package_dir = Path(path) / package_name
                if package_dir.exists() and package_dir.is_dir():
                    return str(package_dir / relative_path)

        # Try system ROS installation paths (e.g., /opt/ros/noetic/share/ur_description)
        for ros_distro in [
            "noetic",
            "melodic",
            "kinetic",
            "humble",
            "foxy",
            "galactic",
        ]:
            system_path = Path(f"/opt/ros/{ros_distro}/share/{package_name}")
            if system_path.exists() and system_path.is_dir():
                return str(system_path / relative_path)

        # Try /usr/share/ros-* paths (some package managers install here)
        for ros_share_path in Path("/usr/share").glob(f"ros-*-{package_name}"):
            if ros_share_path.exists() and ros_share_path.is_dir():
                return str(ros_share_path / relative_path)

        # If we couldn't find it, raise an error
        raise ValueError(
            f"Could not find ROS package '{package_name}' for path '{relative_path}'. "
            f"Searched in: {current_dir / package_name}, {parent_dir / package_name}, "
            f"ROS_PACKAGE_PATH, and system ROS installations. "
            f"Make sure the package is available or install ROS."
        )


def resolve_find_path(path_str):
    """Resolve $(find package_name)/path to actual file path.

    Args:
        path_str (str): Path string that may contain $(find package_name)/path.

    Returns:
        str: Resolved path.
    """
    # Match $(find package_name)/path
    find_pattern = r"\$\(find\s+(\w+)\)(.*)"
    match = re.match(find_pattern, path_str)

    if match:
        package_name = match.group(1)
        relative_path = match.group(2).lstrip("/")
        return _resolve_package_path(package_name, relative_path)
    else:
        # Not a $(find ...) path, return as-is (may be absolute or relative)
        return path_str


def replace_find(match):
    """Callback function for re.sub to replace $(find ...) patterns in text.

    Args:
        match: Regex match object.

    Returns:
        str: Resolved path.
    """
    package_name = match.group(1)
    relative_path = match.group(2).lstrip("/")

    try:
        return _resolve_package_path(package_name, relative_path)
    except ValueError:
        # If we couldn't resolve it, leave it as-is (will cause error later)
        return match.group(0)


def _recursively_preprocess_xacro_file(
    file_path, temp_files, processed_files, depth=0, max_depth=20
):
    """Recursively preprocess a xacro file and all its nested includes.

    Args:
        file_path (str): Path to the xacro file to preprocess.
        temp_files (list): List to collect all temp file paths for cleanup.
        processed_files (dict): Dict mapping original paths to temp paths to avoid reprocessing.
        depth (int): Current recursion depth.
        max_depth (int): Maximum recursion depth.

    Returns:
        str: Path to the preprocessed temporary file.
    """
    if depth > max_depth:
        raise ValueError(
            f"Maximum include depth {max_depth} exceeded in xacro preprocessing."
        )

    # Normalize path to avoid duplicate processing
    file_path = os.path.abspath(file_path)

    # If we've already processed this file, return the cached temp path
    if file_path in processed_files:
        return processed_files[file_path]

    # Read the file content
    with open(file_path, "r") as f:
        content = f.read()

    # Find all <xacro:include> directives and recursively preprocess them
    include_pattern = r'<xacro:include\s+filename\s*=\s*["\']([^"\']+)["\']\s*/>'

    def replace_include_with_preprocessed(match):
        include_path = match.group(1)

        # Resolve the include path (may contain $(find ...))
        resolved_include_path = resolve_find_path(include_path)

        # Recursively preprocess the included file
        preprocessed_include_path = _recursively_preprocess_xacro_file(
            resolved_include_path, temp_files, processed_files, depth + 1, max_depth
        )

        # Return the include directive with the preprocessed path
        return f'<xacro:include filename="{preprocessed_include_path}" />'

    # Replace all include directives with preprocessed paths
    content = re.sub(include_pattern, replace_include_with_preprocessed, content)

    # Preprocess all $(find ...) commands in the content
    content = preprocess_find_commands(content)

    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".urdf.xacro", text=True)
    with os.fdopen(temp_fd, "w") as temp_file:
        temp_file.write(content)

    temp_files.append(temp_path)
    processed_files[file_path] = temp_path

    return temp_path


def parse_and_compile_urdf(d, max_runs=10, compare_existing=True):
    """Parse and compile a URDF from a xacro'd URDF file.

    Args:
        d (dict): Dictionary with "includes" (list) and optionally "args" (dict) keys. Also used for output path via parse_ros_path.
        max_runs (int): Maximum number of xacro processing iterations.
        compare_existing (bool): If True, compare with existing file before writing.

    Returns:
        str: Path to compiled URDF file.

    Raises:
        ValueError: If URDF file does not converge after max_runs iterations.
    """
    # Recursively preprocess all included files and their nested includes
    temp_files = []
    processed_files = {}  # Cache to avoid reprocessing the same file
    preprocessed_includes = []

    for incl in d["includes"]:
        # Resolve the include path itself
        resolved_path = resolve_find_path(incl)

        # Recursively preprocess this file and all its nested includes
        preprocessed_path = _recursively_preprocess_xacro_file(
            resolved_path, temp_files, processed_files
        )
        preprocessed_includes.append(preprocessed_path)

    s = """
    <?xml version="1.0" ?>
    <robot name="thing" xmlns:xacro="http://www.ros.org/wiki/xacro">
    """.strip()
    for incl_path in preprocessed_includes:
        s += xacro_include(incl_path)
    s += "</robot>"

    doc = xacro.parse(s)

    # Preprocess the parsed XML document to catch any remaining $(find ...) commands
    # This handles cases where $(find ...) appears in attributes or macro arguments
    xml_str = doc.toxml()
    preprocessed_xml = preprocess_find_commands(xml_str)
    if preprocessed_xml != xml_str:
        # Re-parse if we made changes
        doc = xacro.parse(preprocessed_xml)

    s1 = doc.toxml()

    # xacro args - also resolve $(find ...) in transform_params if present
    mappings = d["args"] if "args" in d else {}
    if "transform_params" in mappings:
        mappings["transform_params"] = resolve_find_path(mappings["transform_params"])

    # keep processing until a fixed point is reached
    run = 1
    while run < max_runs:
        # Preprocess the document before each process_doc call to catch any $(find ...)
        # that might have been introduced during processing
        xml_str = doc.toxml()
        preprocessed_xml = preprocess_find_commands(xml_str)
        if preprocessed_xml != xml_str:
            doc = xacro.parse(preprocessed_xml)

        xacro.process_doc(doc, mappings=mappings)
        s2 = doc.toxml()
        if s1 == s2:
            break
        s1 = s2
        run += 1

    if run == max_runs:
        # Clean up temp files before raising
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise ValueError("URDF file did not converge.")

    # write the final document to a file for later consumption
    output_path = parse_ros_path(d, as_string=False)

    # make sure path exists
    if not output_path.parent.exists():
        output_path.parent.mkdir()

    text = doc.toprettyxml(indent="  ")

    # if the full path already exists, we can check if the contents are the
    # same to avoid writing it if it hasn't changed. This avoids some race
    # conditions if the file is being compiled by multiple processes
    # concurrently.
    if output_path.exists() and compare_existing:
        with open(output_path) as f:
            text_current = f.read()
        if text_current == text:
            print("URDF files are the same - not writing.")
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return output_path.as_posix()
        else:
            print("URDF files are not the same - writing.")

    with open(output_path, "w") as f:
        f.write(text)

    # Clean up temp files
    for temp_path in temp_files:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return output_path.as_posix()
