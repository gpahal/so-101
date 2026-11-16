"""What every model is asked: the scene, the moment, the question and the answer format.

Chat models (Claude, GPT, Gemini, Gemini Robotics-ER) get SYSTEM plus `question(moment)` and must answer
with JSON matching ANSWER_SCHEMA. Decision models (Jev, Gemini Decision 1, OpenAI Decisions) can't write
text, so they get the same context as input and two closed questions instead: WRONG_QUESTION (yes/no,
answered with a probability) and CATEGORY_QUESTION (a choice over CATEGORIES). Every request says which
images it carries, in order (`describe`), and labels each one (`label`).
"""

import re

TASK = "Pick up the red block and put it in the cylindrical container"

# When the frame was taken, in the words the model sees. The controller always knows which of these it
# is at, so telling the model is realistic, and without it an end frame looks like a start frame.
MOMENTS = {
    "approach": "The arm is moving towards the block to pick it up.",
    "after_close": "About 1 second ago the gripper closed: the robot just tried to grasp the block.",
    "after_release": "About 1.5 seconds ago the gripper opened: the robot just tried to drop the block into the container.",
    "end": "The attempt is over. This is its last frame; the robot will do nothing more.",
    # The supervisor's watch check, asked once when a stuck rule (`grasp.detector`) or a third miss holds
    # the arm, and `so101-failbench candidates`' frame where a rule fired. The watch check adds the same
    # cameras from `lookback_s` earlier, named like `top_2s_ago`.
    "stuck": ("The arm may be stuck: for several seconds it has barely moved, hovered with the jaws open, or "
              "closed and opened again without carrying the block. Frames named like top_2s_ago show the same "
              "camera from that long before."),
}

# What can have gone wrong. `none` means nothing is wrong.
CATEGORIES = {
    "none": "Nothing is wrong for this moment.",
    "missed_grasp": "The gripper closed or is closing but the block is not between the jaws; it is still on the table.",
    "bad_grasp": "The block is in the jaws but held insecurely: by an edge or corner, tilted or half out.",
    "dropped": "The block fell out of the gripper and lies somewhere other than the container.",
    "missed_jar": "The block was released but is outside the container: beside it, against it or on its rim.",
    "still_holding": "The robot should have released the block by now but it is still in the gripper.",
    "not_picked": "The attempt is over and the block was never picked up; it is still on the table.",
    "knocked_over": "The arm tipped the block over or pushed it away.",
    "wrong_target": "The gripper is grasping or reaching for something that is not the block.",
    "other": "Something else is clearly wrong.",
}

# Categories that are hard to tell apart from two frames, scored as one family in "category right":
# the labellers themselves split missed versus bad grasps, and dropped versus beside the jar.
FAMILIES = {"none": "none", "missed_grasp": "grasp", "bad_grasp": "grasp", "dropped": "not_in_jar",
            "missed_jar": "not_in_jar", "not_picked": "not_in_jar", "knocked_over": "not_in_jar",
            "still_holding": "still_holding", "wrong_target": "wrong_target", "other": "other"}

SYSTEM = f"""You supervise a small robot arm (an SO-101) doing one task: "{TASK}".
The block is a red 2x2 toy brick. The container is a clear, amber-tinted cylindrical jar, open side up.
It has two cameras, and each request says which frames it carries, in order:
- the TOP camera: a fixed view from above the table, showing the arm, the block and the jar.
- the WRIST camera: mounted on the gripper; its two white jaws are at the bottom of the image.
Judge only the state of the task now, in the newest frames; frames from earlier, when there are any, show
how it got there. Is something wrong for this moment of the attempt?
Ignore lighting, colour casts, motion blur and whether the table has a black mat; the jar may be in a
different place than usual and the block may start on its side, and that alone is not wrong.
Categories:
""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())


def question(moment: str | None) -> str:
    """The user turn for a chat model. `moment=None` leaves the moment out (the --no-moment variant)."""
    when = f"Moment: {MOMENTS[moment]}\n" if moment else "Moment: unknown.\n"
    return (when + "Is something wrong at this instant? First describe in one sentence where the block is "
            "and what the gripper is doing, then answer. p_wrong is your probability, from 0 to 1, that "
            "something is wrong.")


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "One sentence: where the block is and what the gripper is doing."},
        "wrong": {"type": "boolean", "description": "True if something is wrong at this instant."},
        "p_wrong": {"type": "number", "description": "Probability from 0 to 1 that something is wrong."},
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["observation", "wrong", "p_wrong", "category"],
    "additionalProperties": False,
}

# For decision models: the same context, as plain input text, then two questions.
def decision_context(moment: str | None, images=("top", "wrist")) -> str:
    """The decision models' context for a request carrying `images` (as `describe` takes them)."""
    return SYSTEM + "\n\n" + (f"Moment: {MOMENTS[moment]}" if moment else "Moment: unknown.") + \
        "\n" + describe(images)


ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
_EARLIER = re.compile(r"^(?P<camera>.+?)_(?P<seconds>\d+(?:_\d+)?)s_ago$")


def _parts(name: str) -> tuple[str, str | None]:
    """A frame's camera, upper-cased, and how long ago it was taken (`2.5 s`), or None for now."""
    m = _EARLIER.match(name)
    return (m["camera"].upper(), m["seconds"].replace("_", ".") + " s") if m else (name.upper(), None)


def label(name: str) -> str:
    """What an image is called where a request shows it: `TOP camera`, or `TOP camera, 2 s ago` for the
    watch check's `top_2s_ago`."""
    camera, ago = _parts(name)
    return f"{camera} camera" + (f", {ago} ago" if ago else "")


def describe(images) -> str:
    """The sentence telling a model which images a request carries, in order. `images` is the request's
    (name, JPEG bytes) pairs, or their names: `top`, `wrist`, and for the watch check the same cameras
    from earlier, named like `top_2s_ago`. Every adapter and the supervisor's checks say it the same way:
    "The image is the TOP camera." for one, "The first image is the TOP camera, the second the WRIST
    camera." for two, and with earlier frames each says when: "... the TOP camera now, ... the third the
    TOP camera 2 s ago, ...".
    """
    names = [image[0] if isinstance(image, tuple) else image for image in images]
    if not names:
        return "The request carries no image."
    if len(names) > len(ORDINALS):
        raise ValueError(f"describe names up to {len(ORDINALS)} images, not {len(names)}")
    parts = [_parts(name) for name in names]
    timed = any(ago for _, ago in parts)
    said = [f"the {camera} camera" + ((f" {ago} ago" if ago else " now") if timed else "") for camera, ago in parts]
    if len(said) == 1:
        return f"The image is {said[0]}."
    return "The first image is " + ", ".join([said[0], *(f"the {ORDINALS[i]} {s}" for i, s in enumerate(said[1:], 1))]) + "."


WRONG_QUESTION = "Is something wrong with the task at this instant, for this moment of the attempt?"
CATEGORY_QUESTION = "Which best describes the state of the task at this instant?"
