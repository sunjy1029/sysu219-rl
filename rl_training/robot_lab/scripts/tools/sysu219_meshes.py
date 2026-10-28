"""Derive importable SYSU219 geometry without modifying authoritative ROS files.

The current ImageToStl COLLADA exporter writes accessor count as scalar count.
Accept that known defect, validate every triangle index, and preserve coordinates.
Other COLLADA layouts/transforms are rejected instead of silently flattened.
"""

import io
import json
import xml.etree.ElementTree as ET

import numpy as np


FOOT_MASS_KG = 1.0e-6  # 1 mg per fixed foot: avoid automatic 33.5 g sphere mass.
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dae_to_obj(data):
    root = ET.fromstring(data)
    require(root.findtext("c:asset/c:up_axis", namespaces=NS) == "Z_UP", "Expected Z_UP")
    require(root.find("c:asset/c:unit", NS) is None, "Unexpected COLLADA unit conversion")
    require(root.find(".//c:init_from", NS) is None, "External texture dependency")
    for node in root.findall(".//c:node/*", NS):
        kind = node.tag.split("}")[-1]
        require(kind not in ("translate", "rotate", "scale", "instance_node"), "Unsupported scene transform")
        if kind == "matrix":
            require(np.array_equal(np.fromstring(node.text, sep=" ").reshape(4, 4), np.eye(4)), "Nonidentity scene transform")
    geometries = root.findall("c:library_geometries/c:geometry", NS)
    references = [g.get("url").removeprefix("#") for g in root.findall(".//c:instance_geometry", NS)]
    require(sorted(references) == sorted(g.get("id") for g in geometries), "Unexpected geometry instances")
    out = io.StringIO()
    out.write("# Derived from ROS DAE: unchanged vertices, triangles and URDF mesh scale.\n")
    offset = 0
    for geometry in geometries:
        mesh = geometry.find("c:mesh", NS)
        require({child.tag.split("}")[-1] for child in mesh} <= {"source", "vertices", "triangles"}, "Unsupported primitive")
        for number, triangles in enumerate(mesh.findall("c:triangles", NS)):
            inputs = triangles.findall("c:input", NS)
            require(len(inputs) == 1 and inputs[0].get("semantic") == "VERTEX" and inputs[0].get("offset") == "0", "Unexpected triangle inputs")
            vertex_id = inputs[0].get("source").removeprefix("#")
            vertex = mesh.find(f"c:vertices[@id='{vertex_id}']", NS)
            vinputs = vertex.findall("c:input", NS)
            require(len(vinputs) == 1 and vinputs[0].get("semantic") == "POSITION", "Unexpected vertex inputs")
            source_id = vinputs[0].get("source").removeprefix("#")
            position = mesh.find(f"c:source[@id='{source_id}']", NS)
            accessor = position.find("c:technique_common/c:accessor", NS)
            require(accessor.get("stride") == "3" and int(accessor.get("offset", "0")) == 0, "Unexpected vertex stride")
            array = position.find("c:float_array", NS)
            vertices = np.fromstring(array.text, sep=" ").reshape(-1, 3)
            require(vertices.size == int(array.get("count")), "Float count mismatch")
            require(int(accessor.get("count")) in (len(vertices), vertices.size), "Unknown accessor count defect")
            require(np.isfinite(vertices).all(), "Nonfinite vertices")
            faces = np.fromstring(triangles.findtext("c:p", namespaces=NS), sep=" ", dtype=np.int64).reshape(-1, 3)
            require(len(faces) == int(triangles.get("count")), "Triangle count mismatch")
            require(faces.min() >= 0 and faces.max() < len(vertices), "Triangle outside vertex array")
            out.write(f"o {geometry.get('id')}_{number}\n")
            np.savetxt(out, vertices, fmt="v %.17g %.17g %.17g")
            np.savetxt(out, faces + offset + 1, fmt="f %d %d %d")
            offset += len(vertices)
    return out.getvalue().encode()


def build_isaac_files(files):
    root = ET.fromstring(files["urdf/robot.urdf"])
    wrapper = root.find("joint[@name='floating_base']")
    require(wrapper is not None and wrapper.get("type") == "fixed", "Unexpected base wrapper")
    require(wrapper.find("origin").get("xyz") == "0 0 0" and wrapper.find("origin").get("rpy") == "0 0 0", "Nonidentity base wrapper")
    require(root.find("link[@name='base']/inertial") is None, "Base wrapper has inertia")
    require(root.find("link[@name='base']/collision") is None, "Base wrapper has collision")
    root.remove(wrapper)
    root.remove(root.find("link[@name='base']"))
    derived = {}
    for mesh in root.findall(".//mesh"):
        source_name = mesh.get("filename").removeprefix("../")
        target_name = source_name.replace("meshes/", "meshes_obj/").removesuffix(".dae") + ".obj"
        if target_name not in derived:
            derived[target_name] = dae_to_obj(files[source_name])
        mesh.set("filename", "../" + target_name)
    for leg in ("FR", "FL", "RR", "RL"):
        foot = root.find(f"link[@name='{leg}_foot']")
        require(foot.find("inertial") is None, "Source foot mass must be reviewed before regularization")
        radius = float(foot.find("collision/geometry/sphere").get("radius"))
        inertia = ET.SubElement(foot, "inertial")
        ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertia, "mass", value=str(FOOT_MASS_KG))
        moment = str(0.4 * FOOT_MASS_KG * radius**2)
        ET.SubElement(inertia, "inertia", ixx=moment, iyy=moment, izz=moment, ixy="0", ixz="0", iyz="0")
    ET.indent(root, space="  ")
    derived["urdf/robot_isaac.urdf"] = ("<?xml version=\"1.0\"?>\n<!-- Derived import view; see README for root and fixed-foot regularization. -->\n" + ET.tostring(root, encoding="unicode") + "\n").encode()
    manifest = json.loads(files["model_parameters.json"])
    manifest["isaac_import"] = dict(root_link="trunk", mesh_format="OBJ", foot_regularization_mass_kg=FOOT_MASS_KG,
                                    expected_total_mass_kg=manifest["total_mass_kg"] + 4 * FOOT_MASS_KG)
    derived["model_parameters.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return derived
