""" top level run script """
import os
import subprocess


def run():
    """ basic run function """
    session_ids = [1089296550, 1093642839, 1096620314, 1108334384, 1118324999, 1118327332, 1139846596]

    for session_id in session_ids:
        os.environ["session_id"] = str(session_id)
        command = [
                "jupyter",
                "nbconvert",
                "--execute",
                "~/capsule/code/ephys_plotter.ipynb",
                "--to",
                "notebook",
                "--output",
                f"~/capsule/results/{session_id}_report.ipynb",
            ]
        # subprocess.run(command, check=True)

if __name__ == "__main__": run()