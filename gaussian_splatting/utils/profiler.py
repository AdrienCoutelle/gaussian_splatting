import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable

from gaussian_splatting.utils.logger import Logger

logger = Logger("PROFILER")


class Profiler:
    _timings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    _enabled: bool = True

    @classmethod
    def enable(cls) -> None:
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()

    @classmethod
    def profile(
        cls,
        target: Callable | type,
    ) -> Callable | type:
        if isinstance(target, type):
            return cls._profile_class(target)
        else:
            return cls._profile_method(target)

    @classmethod
    def _profile_method(
        cls,
        func: Callable,
        class_name: str | None = None,
    ) -> Callable:
        @wraps(func)
        def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not cls._enabled:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time

                # Determine class name
                if class_name is not None:
                    target_class = class_name
                elif args and hasattr(args[0], "__class__"):
                    # Check if it's a classmethod (args[0] is a class)
                    if isinstance(args[0], type):
                        target_class = args[0].__name__
                    else:
                        target_class = args[0].__class__.__name__
                else:
                    target_class = "Functions"

                method_name = func.__name__
                cls._timings[target_class][method_name].append(duration)

        return wrapper

    @classmethod
    def _profile_class(
        cls,
        target_class: type,
    ) -> type:
        class_name = target_class.__name__

        for attr_name in dir(target_class):
            if attr_name.startswith("__"):
                continue

            raw = None
            for klass in target_class.__mro__:
                if attr_name in klass.__dict__:
                    raw = klass.__dict__[attr_name]
                    break

            if isinstance(raw, staticmethod):
                wrapped = cls._profile_method(raw.__func__, class_name=class_name)
                setattr(target_class, attr_name, staticmethod(wrapped))
                continue

            attr = getattr(target_class, attr_name)

            if callable(attr):
                setattr(
                    target_class,
                    attr_name,
                    cls._profile_method(attr, class_name=class_name),
                )

        return target_class

    @classmethod
    def print_stats(cls) -> None:
        if len(cls._timings) == 0:
            logger.info("No profiling data available.")
            return

        message = "\n" + "=" * 80
        message += "\nPROFILING STATISTICS"
        message += "\n" + "=" * 80
        message += "\n"

        for class_name in sorted(cls._timings.keys()):
            methods = cls._timings[class_name]

            message += "\n" + "-" * 80
            message += f"\n• {class_name}:"
            message += "\n"

            method_stats = [
                (
                    method_name,
                    timings,
                    sum(timings),
                    len(timings),
                    sum(timings) / len(timings),
                    min(timings),
                    max(timings),
                )
                for method_name, timings in methods.items()
            ]  # fmt:skip

            method_stats.sort(key=lambda x: x[2], reverse=True)

            message += (
                "\n" + f"{'Method':<30} {'Calls':>8} {'Total (s)':>12} {'Avg (s)':>12} {'Min (s)':>12} {'Max (s)':>12}"
            )
            message += "\n" + "- " * 40

            for method_name, _, total, count, avg, min_time, max_time in method_stats:
                message += (
                    "\n"
                    + f"{method_name:<30} {count:>8} {total:>12.6f} {avg:>12.6f} {min_time:>12.6f} {max_time:>12.6f}"
                )

        message += "\n" + "=" * 80 + "\n"

        logger.info(message)


profile = Profiler.profile
