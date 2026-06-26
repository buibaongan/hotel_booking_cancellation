import argparse

from main import BookingCancellationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Booking cancellation pipeline")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--model", default="RandomForest")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    pipeline = BookingCancellationPipeline(args.config)
    if args.preprocess_only:
        pipeline.run_preprocessing()
    else:
        pipeline.run(args.model)


if __name__ == "__main__":
    main()
