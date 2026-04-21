from pathlib import Path
import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".mov"}

def load_squat_videos(squat_folders):
    """
    Load squat video metadata from one or more squat folders.

    Args:
        squat_folders (list[str] | list[Path]):
            List of folder paths that contain squat videos.

    Returns:
        pd.DataFrame:
            Columns:
            - video_path
            - file_name
            - exercise_label
            - dataset_source
            - extension
    """
    records = []

    for folder in squat_folders:
        folder = Path(folder)

        if not folder.exists():
            print(f"Warning: folder not found -> {folder}")
            continue

        dataset_source = folder.parent.name  # parent folder name as source

        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append({
                    "video_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "exercise_label": "squat",
                    "dataset_source": dataset_source,
                    "extension": file_path.suffix.lower()
                })

    df = pd.DataFrame(records)

    if df.empty:
        print("No squat videos found.")
        return df

    df = df.sort_values(by=["dataset_source", "file_name"]).reset_index(drop=True)

    print(f"\nTotal squat videos found: {len(df)}")
    print("\nCount by dataset source:")
    print(df["dataset_source"].value_counts())

    return df


if __name__ == "__main__":
    squat_folders = [
        r"workoutfitness-video/squat",
        r"real-time-exercise-recognition-dataset/final_kaggle_with_additional_video/squat"
    ]

    squat_df = load_squat_videos(squat_folders)

    if not squat_df.empty:
        print("\nFirst 5 rows:")
        print(squat_df.head())

        # Save metadata to CSV
        squat_df.to_csv("squat_video_metadata.csv", index=False)
        print("\nSaved metadata to squat_video_metadata.csv")