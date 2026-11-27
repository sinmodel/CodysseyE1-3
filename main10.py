import inspect
import json
import math
import os
import re
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPSILON = 1e-9
REPEAT_COUNT = 10
MAX_PATTERN_SIZE = 25
SIZES = (3, 5, 13, 25)
SIZE_KEY_PATTERN = re.compile(r"^size_(\d+)_(.+)$")


class InputCancelled(Exception):
    pass


WORK_UNITS = [
    ("개발 환경 및 프로젝트 기초", "완료"),
    ("행렬 검증 및 데이터 구조", "완료"),
    ("MAC 연산과 epsilon 판정", "완료"),
    ("라벨 정규화", "완료"),
    ("사용자 입력 모드", "완료"),
    ("JSON 데이터 분석", "완료"),
    ("성능 분석", "완료"),
    ("통합 실행 검증", "완료"),
    ("패턴 생성기", "완료"),
    ("메모리 접근 비교", "완료"),
]


def validate_matrix(matrix, expected_size=None):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("행렬은 비어 있지 않은 2차원 배열이어야 합니다.")

    size = len(matrix)
    if expected_size is not None and size != expected_size:
        raise ValueError(
            "행렬 크기 불일치: 예상 {}x{}, 실제 {}행입니다.".format(
                expected_size, expected_size, size
            )
        )

    for row_index, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != size:
            actual_columns = len(row) if isinstance(row, list) else 0
            raise ValueError(
                "{}행의 열 개수가 올바르지 않습니다: {}개 (예상 {}개).".format(
                    row_index, actual_columns, size
                )
            )
        for column_index, value in enumerate(row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "{}행 {}열 값이 숫자가 아닙니다.".format(row_index, column_index)
                )
            if not math.isfinite(value):
                raise ValueError(
                    "{}행 {}열 값은 유한한 숫자여야 합니다.".format(
                        row_index, column_index
                    )
                )
    return size


def _validate_matrix_position(matrix, row, column):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("행렬은 비어 있지 않은 2차원 배열이어야 합니다.")
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("행 위치는 정수여야 합니다.")
    if isinstance(column, bool) or not isinstance(column, int):
        raise ValueError("열 위치는 정수여야 합니다.")
    if not 0 <= row < len(matrix):
        raise ValueError("행 위치가 행렬 범위를 벗어났습니다.")
    if not isinstance(matrix[row], list) or not 0 <= column < len(matrix[row]):
        raise ValueError("열 위치가 행렬 범위를 벗어났습니다.")


def set_matrix_value(matrix, row, column, value):
    _validate_matrix_position(matrix, row, column)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("행렬 값은 숫자여야 합니다.")
    if not math.isfinite(value):
        raise ValueError("행렬 값은 유한한 숫자여야 합니다.")
    matrix[row][column] = value


def get_matrix_value(matrix, row, column):
    _validate_matrix_position(matrix, row, column)
    return matrix[row][column]


def mac(pattern, filter_matrix):
    size = validate_matrix(pattern)
    validate_matrix(filter_matrix, size)
    score = 0.0
    for row in range(size):
        for column in range(size):
            score += get_matrix_value(pattern, row, column) * get_matrix_value(
                filter_matrix, row, column
            )
    return score


def normalize_label(value):
    if not isinstance(value, str):
        return None
    label = value.strip().casefold()
    if label in ("cross", "+"):
        return "Cross"
    if label == "x":
        return "X"
    return None


def normalize_filter_set(filter_set, size):
    if not isinstance(filter_set, dict):
        raise ValueError("필터 묶음은 객체여야 합니다.")

    normalized_filters = {}
    for raw_label, matrix in filter_set.items():
        label = normalize_label(raw_label)
        if label not in ("Cross", "X"):
            continue
        if label in normalized_filters:
            raise ValueError("중복된 표준 필터 라벨입니다: {}".format(label))
        validate_matrix(matrix, size)
        normalized_filters[label] = matrix

    for label in ("Cross", "X"):
        if label not in normalized_filters:
            raise ValueError("필수 필터 라벨이 없습니다: {}".format(label))
    return normalized_filters


STANDARD_LABELS = ("Cross", "X")


def normalize_expected_label(value):
    # expected 값: '+' -> Cross, 'x' -> X (이미 표준 라벨인 Cross도 허용)
    if not isinstance(value, str):
        return None
    label = value.strip().casefold()
    if label in ("+", "cross"):
        return "Cross"
    if label == "x":
        return "X"
    return None


def normalize_filter_key(key):
    # filter 키: 'cross' -> Cross, 'x' -> X
    if not isinstance(key, str):
        return None
    label = key.strip().casefold()
    if label == "cross":
        return "Cross"
    if label == "x":
        return "X"
    return None


def judge_labels(cross_score, x_score, epsilon=EPSILON):
    if abs(cross_score - x_score) < epsilon:
        return "UNDECIDED"
    return "Cross" if cross_score > x_score else "X"


def judge(score_a, score_b, epsilon=EPSILON):
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"
    return "A" if score_a > score_b else "B"


def input_with_cancel(prompt, allow_cancel=False):
    while True:
        value = input(prompt)
        if not allow_cancel or value.strip().casefold() != "q":
            return value

        while True:
            confirmation = input(
                "입력을 취소하고 메뉴로 돌아갈까요? (y/n): "
            ).strip().casefold()
            if confirmation in ("y", "yes", "예"):
                raise InputCancelled
            if confirmation in ("n", "no", "아니오", "아니요"):
                print("입력을 계속합니다.")
                break
            print("y 또는 n으로 답해 주세요.")


def read_matrix(label, size, allow_cancel=False):
    while True:
        print("{} ({}줄 입력, 공백 구분)".format(label, size))
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        try:
            input_rows = []
            for row_number in range(1, size + 1):
                input_rows.append(
                    input_with_cancel(
                        "{}행: ".format(row_number), allow_cancel
                    ).split()
                )

            for row_number, values in enumerate(input_rows, start=1):
                if len(values) != size:
                    raise ValueError
                row = [float(value) for value in values]
                if not all(math.isfinite(value) for value in row):
                    raise ValueError
                for column, value in enumerate(row):
                    set_matrix_value(matrix, row_number - 1, column, value)
            validate_matrix(matrix, size)
            return matrix
        except (ValueError, OverflowError):
            print(
                "입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(
                    size
                )
            )


def validate_input_matrix(input_rows, expected_size):
    errors = []
    if len(input_rows) != expected_size:
        errors.append(
            "행 수 불일치: 입력 {}행, 예상 {}행".format(
                len(input_rows), expected_size
            )
        )

    matrix = []
    for row_number, values in enumerate(input_rows, start=1):
        if len(values) != expected_size:
            errors.append(
                "{}행 숫자 개수 불일치: 입력 {}개, 예상 {}개".format(
                    row_number, len(values), expected_size
                )
            )
            continue
        try:
            row = [float(value) for value in values]
        except (ValueError, OverflowError):
            errors.append("{}행에 숫자가 아닌 값이 있습니다.".format(row_number))
            continue
        if not all(math.isfinite(value) for value in row):
            errors.append(
                "{}행에 유한하지 않은 숫자가 있습니다.".format(row_number)
            )
            continue
        matrix.append(row)

    return matrix, errors


def confirm_matrix_retry(allow_cancel=False):
    while True:
        answer = input_with_cancel(
            "행렬을 처음부터 다시 입력하겠어요? (y/n): ",
            allow_cancel,
        ).strip().casefold()
        if answer in ("y", "yes", "예"):
            return True
        if answer in ("n", "no", "아니요", "아니오"):
            return False
        print("y 또는 n으로 답해 주세요.")


def read_matrix_until_confirmed(label, expected_size, allow_cancel=False):
    while True:
        input_rows = []
        print("{} 행을 한 줄씩 입력하세요.".format(label))
        while True:
            row_input = input_with_cancel(
                "{}행: ".format(len(input_rows) + 1), allow_cancel
            ).strip()
            input_rows.append(row_input.split())
            if len(input_rows) < expected_size:
                continue

            while True:
                answer = input_with_cancel(
                    "입력을 여기서 멈추고 검사할까요? (y/n): ", allow_cancel
                ).strip().casefold()
                if answer in ("y", "yes", "예"):
                    finished = True
                    break
                if answer in ("n", "no", "아니요", "아니오"):
                    finished = False
                    break
                print("y 또는 n으로 답해 주세요.")
            if finished:
                break

        matrix, errors = validate_input_matrix(input_rows, expected_size)

        if errors:
            print(
                "입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(
                    expected_size
                )
            )
            for error in errors:
                print("- {}".format(error))
            if not confirm_matrix_retry(allow_cancel):
                raise InputCancelled
            continue

        validate_matrix(matrix, expected_size)
        return matrix


def load_json(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "data.json")
    with open(path, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조는 객체여야 합니다.")
    if not isinstance(data.get("filters"), dict):
        raise ValueError("'filters'는 객체여야 합니다.")
    if not isinstance(data.get("patterns"), dict):
        raise ValueError("'patterns'는 객체여야 합니다.")
    return data


def analyze_json(data):
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조는 객체여야 합니다.")
    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("'filters'와 'patterns'는 모두 객체여야 합니다.")

    filters_by_size = {}
    for size in (5, 13, 25):
        filter_key = "size_{}".format(size)
        if filter_key not in filters:
            print("필터 로드 실패: {}가 없습니다.".format(filter_key))
            continue
        try:
            filter_set = normalize_filter_set(filters[filter_key], size)
        except (ValueError, TypeError) as error:
            print("필터 로드 실패: {} - {}".format(filter_key, error))
            continue
        filters_by_size[size] = filter_set
        print("[완료] {} 필터 로드 완료 (Cross, X)".format(filter_key))

    total = 0
    passed = 0
    failures = []

    for case_id, case in patterns.items():
        total += 1
        print("\n- -- {} ---".format(case_id))
        try:
            if not isinstance(case_id, str):
                raise ValueError("케이스 식별자가 문자열이 아닙니다.")
            match = SIZE_KEY_PATTERN.fullmatch(case_id)
            if match is None:
                raise ValueError("케이스 키 형식이 size_{N}_{idx}와 다릅니다.")
            size = int(match.group(1))
            if size not in (5, 13, 25):
                raise ValueError("지원하지 않는 패턴 크기입니다: {}".format(size))
            if not isinstance(case, dict):
                raise ValueError("케이스 구조가 객체여야 합니다.")
            if size not in filters_by_size:
                raise ValueError("size_{} 필터를 사용할 수 없습니다.".format(size))

            pattern = case.get("input")
            validate_matrix(pattern, size)
            filter_set = filters_by_size[size]
            cross_score = mac(pattern, filter_set["Cross"])
            x_score = mac(pattern, filter_set["X"])

            expected = normalize_label(case.get("expected"))
            if expected is None:
                raise ValueError("expected 라벨을 Cross 또는 X로 변환할 수 없습니다.")

            result = judge_labels(cross_score, x_score)
            status = "PASS" if result == expected else "FAIL"
            print("Cross 점수: {}".format(cross_score))
            print("X 점수: {}".format(x_score))
            print(
                "판정: {} | expected: {} | {}".format(
                    result, expected, status
                )
            )

            if status == "PASS":
                passed += 1
            else:
                reason = (
                    "동점(UNDECIDED) 처리 규칙"
                    if result == "UNDECIDED"
                    else "판정 결과와 expected 불일치"
                )
                failures.append((case_id, reason))
        except (ValueError, TypeError, KeyError) as error:
            reason = "스키마/데이터 오류: {}".format(error)
            print("판정: FAIL")
            print("사유: {}".format(reason))
            failures.append((case_id, reason))

    print("\n결과 요약")
    print("총 테스트: {}개".format(total))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(total - passed))
    if failures:
        print("실패 케이스:")
        for case_id, reason in failures:
            print("- {}: {}".format(case_id, reason))

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "failures": failures,
    }


def summarize_data_structure(data):
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조는 객체여야 합니다.")

    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("'filters'와 'patterns'는 모두 객체여야 합니다.")

    for size in (5, 13, 25):
        filter_key = "size_{}".format(size)
        print("\n[{}×{} 데이터]".format(size, size))
        if filter_key not in filters:
            print("필터: {}가 없습니다.".format(filter_key))
        else:
            try:
                normalize_filter_set(filters[filter_key], size)
                print("필터: Cross, X (각 {}×{})".format(size, size))
            except (ValueError, TypeError) as error:
                print("필터 구조 오류: {}".format(error))

        cases = []
        for case_id, case in patterns.items():
            match = (
                SIZE_KEY_PATTERN.fullmatch(case_id)
                if isinstance(case_id, str)
                else None
            )
            if match is not None and int(match.group(1)) == size:
                cases.append((case_id, case))

        print("패턴 케이스: {}개".format(len(cases)))
        for case_id, case in cases:
            if not isinstance(case, dict):
                print("- {}: 케이스 구조 오류".format(case_id))
                continue

            try:
                actual_size = validate_matrix(case.get("input"), size)
                input_shape = "{}×{}".format(actual_size, actual_size)
            except (ValueError, TypeError) as error:
                input_shape = "구조 오류 ({})".format(error)

            expected = normalize_label(case.get("expected")) or "알 수 없음"
            print(
                "- {}: 입력 {} / expected {}".format(
                    case_id, input_shape, expected
                )
            )


def timed_average(operation, repeat_count=REPEAT_COUNT):
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise ValueError("반복 횟수는 정수여야 합니다.")
    if repeat_count < 1:
        raise ValueError("반복 횟수는 1 이상이어야 합니다.")
    start = time.perf_counter_ns()
    for _ in range(repeat_count):
        operation()
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / repeat_count / 1_000_000


def generate_pattern(size, label):
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("크기 N은 정수여야 합니다.")
    if not 1 <= size <= MAX_PATTERN_SIZE:
        raise ValueError(
            "크기 N은 1부터 {}까지 입력하세요.".format(MAX_PATTERN_SIZE)
        )

    normalized_label = normalize_label(label)
    if normalized_label is None:
        raise ValueError("패턴 종류는 Cross 또는 X여야 합니다.")

    pattern = []
    center_rows = {(size - 1) // 2, size // 2}
    for row in range(size):
        values = []
        for column in range(size):
            if normalized_label == "Cross":
                value = int(row in center_rows or column in center_rows)
            else:
                value = int(row == column or row + column == size - 1)
            values.append(value)
        pattern.append(values)
    return pattern


def flatten_matrix(matrix, expected_size=None):
    validate_matrix(matrix, expected_size)
    return [value for row in matrix for value in row]


def mac_flat(pattern_flat, filter_flat):
    if not isinstance(pattern_flat, list) or not isinstance(filter_flat, list):
        raise ValueError("1차원 행렬은 리스트여야 합니다.")
    if not pattern_flat or len(pattern_flat) != len(filter_flat):
        raise ValueError("1차원 패턴과 필터의 길이가 같고 비어 있지 않아야 합니다.")

    size = math.isqrt(len(pattern_flat))
    if size * size != len(pattern_flat):
        raise ValueError("1차원 행렬의 길이는 완전제곱수여야 합니다.")

    score = 0.0
    for index in range(len(pattern_flat)):
        pattern_value = pattern_flat[index]
        filter_value = filter_flat[index]
        for value in (pattern_value, filter_value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("1차원 행렬의 각 값은 숫자여야 합니다.")
            if not math.isfinite(value):
                raise ValueError("1차원 행렬의 각 값은 유한한 숫자여야 합니다.")
        score += pattern_value * filter_value
    return score


def make_benchmark_matrices(size):
    return generate_pattern(size, "Cross"), generate_pattern(size, "X")


def print_performance_report():
    print("크기(N×N)\t평균 시간(ms)\t연산 횟수(N²)")
    print("-" * 42)
    for size in SIZES:
        pattern, filter_matrix = make_benchmark_matrices(size)
        average_ms = timed_average(
            lambda: mac(pattern, filter_matrix),
            REPEAT_COUNT,
        )
        print(
            "{}×{}\t{:.6f}\t\t{}".format(
                size, size, average_ms, size * size
            )
        )


def compare_memory_access(sizes=SIZES):
    print("크기(N×N)\t2D 평균(ms)\t1D 평균(ms)\t1D 배율\t연산 횟수(N²)")
    print("-" * 70)
    for size in sizes:
        pattern, filter_matrix = make_benchmark_matrices(size)
        pattern_flat = flatten_matrix(pattern)
        filter_flat = flatten_matrix(filter_matrix)
        average_2d = timed_average(
            lambda: mac(pattern, filter_matrix),
            REPEAT_COUNT,
        )
        average_1d = timed_average(
            lambda: mac_flat(pattern_flat, filter_flat),
            REPEAT_COUNT,
        )
        ratio = average_2d / average_1d if average_1d > 0 else float("inf")
        print(
            "{}×{}\t\t{:.6f}\t{:.6f}\t{:.2f}배\t{}".format(
                size, size, average_2d, average_1d, ratio, size * size
            )
        )


def clear_screen():
    command = "cls" if os.name == "nt" else "clear"
    if os.system(command) != 0:
        print("화면 정리 명령 실행에 실패했습니다.")


def print_user_mode_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용:")
    print("import math")
    print("import time")
    print()
    print("EPSILON = {}".format(EPSILON))
    print("REPEAT_COUNT = {}".format(REPEAT_COUNT))
    print()

    for function in (
        validate_matrix,
        _validate_matrix_position,
        set_matrix_value,
        get_matrix_value,
        mac,
        judge,
        read_matrix,
        timed_average,
        run_user_mode,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("-" * 55)


def print_json_mode_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용:")
    print("import json")
    print("import math")
    print("import os")
    print("import re")
    print()
    print("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
    print("EPSILON = {}".format(EPSILON))
    print('SIZE_KEY_PATTERN = re.compile(r"^size_(\\d+)_(.+)$")')
    print()

    for function in (
        validate_matrix,
        _validate_matrix_position,
        set_matrix_value,
        get_matrix_value,
        mac,
        normalize_label,
        normalize_filter_set,
        judge_labels,
        load_json,
        analyze_json,
        run_json_mode,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("-" * 55)


def print_data_structure_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용:")
    print("import json")
    print("import math")
    print("import os")
    print("import re")
    print()
    print("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
    print('SIZE_KEY_PATTERN = re.compile(r"^size_(\\d+)_(.+)$")')
    print()

    for function in (
        validate_matrix,
        normalize_label,
        normalize_filter_set,
        load_json,
        summarize_data_structure,
        run_data_structure_mode,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("-" * 55)


def print_input_processing_code():
    print("\n" + "-" * 55)
    print("\n핵심 코딩 내용:")
    print("import math")
    print()
    print("matrix, errors = validate_input_matrix(input_rows, 3)")
    print(inspect.getsource(validate_input_matrix).rstrip())
    print("-" * 55)


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(value) for value in row))


def run_user_mode():
    print("문제 내용:")
    print("필터 A, 필터 B, 패턴을 각각 3×3 행렬로 입력하세요.")
    print("각 행에는 숫자 3개를 공백으로 구분해 입력합니다.")
    print("패턴과 두 필터를 비교해 점수가 높은 필터를 판정합니다.")
    filter_a = read_matrix("필터 A", 3)
    filter_b = read_matrix("필터 B", 3)
    print("패턴")
    pattern = read_matrix("패턴", 3)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    result = judge(score_a, score_b)
    average_ms = timed_average(
        lambda: (mac(pattern, filter_a), mac(pattern, filter_b)),
        REPEAT_COUNT,
    )

    print("\nA 점수: {}".format(score_a))
    print("B 점수: {}".format(score_b))
    print("연산 시간(평균/{}회): {:.6f} ms".format(REPEAT_COUNT, average_ms))
    if result == "UNDECIDED":
        print("판정: 판정 불가 (동점)")
    else:
        print("판정: {}".format(result))
    print_user_mode_code()
    return True


def run_json_mode():
    print("문제 내용:")
    print("data.json의 5×5, 13×13, 25×25 패턴을 각 크기의 Cross/X 필터와 비교합니다.")
    print("판정이 expected와 같으면 PASS, 다르면 FAIL로 표시합니다.")
    try:
        data = load_json()
        analyze_json(data)
        print_json_mode_code()
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def run_data_structure_mode():
    print("문제 내용:")
    print("data.json에서 5×5, 13×13, 25×25 필터와 패턴의 구조를 확인합니다.")
    print("행렬 전체 대신 필터 라벨, 케이스 수, 입력 크기, expected를 요약합니다.")
    try:
        data = load_json()
        summarize_data_structure(data)
        print_data_structure_code()
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def run_input_processing_mode():
    print("문제 내용:")
    print("필터 A, 필터 B, 패턴을 각각 3×3 행렬로 입력하세요.")
    print("3행 입력 후 y를 누르면 지금까지 입력한 행을 검사합니다.")
    print("n을 누르면 행을 더 입력하고, 검사할 때 다시 y를 누릅니다.")
    print("3×3은 숫자 9개이며, 입력을 마친 뒤 행 수와 숫자 형식을 검사합니다.")
    print("입력된 행렬을 확인용으로 출력하며, 점수 판정은 하지 않습니다.")
    print("입력 도중 q를 누르면 메뉴로 돌아갈지 확인합니다.")
    print("재입력 예시:")
    print("아래는 설명용이며, 예시를 일부러 잘못 입력할 필요는 없습니다.")
    print("1) 숫자 개수나 형식이 잘못된 3×3 입력:")
    print("        1열  2열  3열  4열")
    print("   1행:  1    0    -    -")
    print("   2행:  0    1    0    1")
    print("   3행:  1    a    0    -")
    print("   -는 값이 없는 칸이고, a는 숫자가 아닌 값입니다.")
    print("   3행 뒤 y라고 답하면 오류를 안내합니다.")
    print("2) 4개 행을 입력하고 행마다 열 개수도 다른 경우:")
    print("        1열  2열  3열  4열")
    print("   1행:  0    0    1    -")
    print("   2행:  1    0    1    0")
    print("   3행:  1    1    -    -")
    print("   4행:  0    1    0    -")
    print("   -는 값이 없는 칸이며, 실제 입력에는 -를 쓰지 않습니다.")
    print("   3행 뒤 n, 4행 뒤 y라고 답하면 행 수와 열 개수 오류를 안내합니다.")

    matrices = []
    try:
        for label in ("필터 A", "필터 B", "패턴"):
            matrices.append(
                (
                    label,
                    read_matrix_until_confirmed(
                        label, 3, allow_cancel=True
                    ),
                )
            )
    except InputCancelled:
        print("입력이 취소되었습니다. 핵심 코딩 내용을 표시합니다.")
        print_input_processing_code()
        return False

    print("\n입력 확인")
    for label, matrix in matrices:
        print("{} 입력 완료:".format(label))
        print_matrix(matrix)

    print_input_processing_code()
    return True



def print_schema_validation_code():
    print("\n" + "-" * 55)
    print("\n핵심 코딩 내용:")
    print("import json")
    print("import math")
    print("import os")
    print("import re")
    print()
    print("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
    print('SIZE_KEY_PATTERN = re.compile(r"^size_(\\d+)_(.+)$")')
    print()
    for function in (
        validate_matrix,
        normalize_label,
        normalize_filter_set,
        load_json,
        analyze_json,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("-" * 55)


def run_schema_validation_mode():
    print("문제 내용:")
    print("data.json에서 filters와 patterns를 읽어옵니다.")
    print("패턴 키에서 크기 N을 추출해 size_N 필터를 선택합니다.")
    print("필터와 패턴의 행렬 크기를 검증하고, 오류가 있으면 해당 케이스만 FAIL 처리합니다.")
    print("오류가 발생해도 프로그램을 종료하지 않고 나머지 케이스를 계속 검사합니다.")
    try:
        data = load_json()
        result = analyze_json(data)
        print_schema_validation_code()
        print("\n[문제 5 결과]")
        print("총 테스트: {}개 / PASS: {}개 / FAIL: {}개".format(
            result["total"], result["passed"], result["failed"]
        ))
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def analyze_normalized_labels(data):
    filters = data.get("filters")
    patterns = data.get("patterns")

    print("\n[1] filter 키 정규화")
    filters_by_size = {}
    for size in (5, 13, 25):
        size_key = "size_{}".format(size)
        filter_set = filters.get(size_key)
        if not isinstance(filter_set, dict):
            print("- {}: 필터가 없습니다.".format(size_key))
            continue

        standard_filters = {}
        for raw_key, matrix in filter_set.items():
            label = normalize_filter_key(raw_key)
            if label is None:
                print("- {}: '{}' -> 변환 불가 (무시)".format(size_key, raw_key))
                continue
            print("- {}: '{}' -> {}".format(size_key, raw_key, label))
            standard_filters[label] = matrix

        missing = [name for name in STANDARD_LABELS if name not in standard_filters]
        if missing:
            print("  필수 표준 라벨이 없습니다: {}".format(", ".join(missing)))
            continue
        try:
            for matrix in standard_filters.values():
                validate_matrix(matrix, size)
        except ValueError as error:
            print("  필터 행렬 오류: {}".format(error))
            continue
        filters_by_size[size] = standard_filters

    print("\n[2] expected 정규화 및 PASS/FAIL (표준 라벨 기준)")
    total = 0
    passed = 0
    failures = []
    for case_id, case in patterns.items():
        total += 1
        try:
            match = SIZE_KEY_PATTERN.fullmatch(case_id)
            if match is None:
                raise ValueError("케이스 키 형식이 size_{N}_{idx}와 다릅니다.")
            size = int(match.group(1))
            if size not in filters_by_size:
                raise ValueError("size_{} 필터를 사용할 수 없습니다.".format(size))
            if not isinstance(case, dict):
                raise ValueError("케이스 구조가 객체여야 합니다.")

            raw_expected = case.get("expected")
            expected = normalize_expected_label(raw_expected)
            if expected is None:
                raise ValueError(
                    "expected '{}'를 Cross 또는 X로 변환할 수 없습니다.".format(
                        raw_expected
                    )
                )

            pattern = case.get("input")
            validate_matrix(pattern, size)
            cross_score = mac(pattern, filters_by_size[size]["Cross"])
            x_score = mac(pattern, filters_by_size[size]["X"])
            result = judge_labels(cross_score, x_score)
            status = "PASS" if result == expected else "FAIL"
            print(
                "- {}: expected '{}' -> {} | Cross={} X={} | 판정 {} | {}".format(
                    case_id, raw_expected, expected,
                    cross_score, x_score, result, status,
                )
            )
            if status == "PASS":
                passed += 1
            else:
                failures.append((case_id, "판정 결과와 expected 불일치"))
        except (ValueError, TypeError, KeyError) as error:
            print("- {}: FAIL ({})".format(case_id, error))
            failures.append((case_id, str(error)))

    print("\n결과 요약")
    print("총 테스트: {}개 / PASS: {}개 / FAIL: {}개".format(
        total, passed, total - passed
    ))
    return {"total": total, "passed": passed, "failed": total - passed}


def print_label_normalization_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용 (이 부분만 복사해도 data.json과 같은 폴더에서 단독 실행됩니다):")
    print("import json")
    print("import math")
    print("import os")
    print("import re")
    print()
    print("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
    print("EPSILON = {}".format(EPSILON))
    print('SIZE_KEY_PATTERN = re.compile(r"^size_(\\d+)_(.+)$")')
    print('STANDARD_LABELS = ("Cross", "X")')
    print()
    for function in (
        validate_matrix,
        _validate_matrix_position,
        set_matrix_value,
        get_matrix_value,
        mac,
        judge_labels,
        load_json,
        normalize_expected_label,
        normalize_filter_key,
        analyze_normalized_labels,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("def main():")
    print("    try:")
    print("        data = load_json()")
    print("        analyze_normalized_labels(data)")
    print("    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:")
    print('        print("실행 오류: {}".format(error))')
    print()
    print()
    print('if __name__ == "__main__":')
    print("    main()")
    print("-" * 55)


def run_label_normalization_mode():
    print("문제 내용:")
    print("프로그램 내부에서는 표준 라벨 Cross, X 두 가지만 사용합니다.")
    print("expected 값: '+' -> Cross, 'x' -> X")
    print("filter 키: 'cross' -> Cross, 'x' -> X")
    print("콘솔 출력과 PASS/FAIL 비교는 표준 라벨(Cross/X) 기준으로 합니다.")

    print("\n[변환 규칙 확인]")
    for raw in ("+", "x"):
        print("expected '{}' -> {}".format(raw, normalize_expected_label(raw)))
    for raw in ("cross", "x"):
        print("filter 키 '{}' -> {}".format(raw, normalize_filter_key(raw)))

    try:
        data = load_json()
        analyze_normalized_labels(data)
        print_label_normalization_code()
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def print_mac_grid(pattern, filter_matrix):
    size = len(pattern)
    products = [
        [pattern[row][column] * filter_matrix[row][column] for column in range(size)]
        for row in range(size)
    ]
    width = max(
        len(str(value))
        for matrix in (pattern, filter_matrix, products)
        for row in matrix
        for value in row
    )
    print("   패턴      ×    필터      =    위치별 곱")
    middle = size // 2
    for row in range(size):
        blocks = []
        for matrix in (pattern, filter_matrix, products):
            blocks.append(
                " ".join(str(value).rjust(width) for value in matrix[row])
            )
        print("  [{}]   {}   [{}]   {}   [{}]".format(
            blocks[0], "×" if row == middle else " ",
            blocks[1], "=" if row == middle else " ",
            blocks[2],
        ))
    print("  위치별 곱을 모두 더한 값 = {}".format(float(sum(sum(row) for row in products))))


def demo_mac_examples():
    examples = [
        (
            "예제 1 (3×3)",
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            5.0,
        ),
        (
            "예제 2 (3×3)",
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 0, -1], [0, 1, 0], [-1, 0, 1]],
            5.0,
        ),
        (
            "예제 3 (2×2, 소수)",
            [[0.5, 1.5], [2.0, 0.0]],
            [[2, 2], [1, 3]],
            6.0,
        ),
    ]
    for title, pattern, filter_matrix, expected_score in examples:
        print("\n[{}]".format(title))
        terms = []
        for row in range(len(pattern)):
            for column in range(len(pattern[row])):
                terms.append(
                    "{}*{}".format(pattern[row][column], filter_matrix[row][column])
                )
        print_mac_grid(pattern, filter_matrix)
        print("위치별 곱: " + " + ".join(terms))
        score = mac(pattern, filter_matrix)
        check = "일치" if abs(score - expected_score) < EPSILON else "불일치"
        print("MAC 점수: {} (손계산 {}) -> {}".format(score, expected_score, check))


def demo_epsilon_policy():
    pairs = [
        ("같은 점수", 5.0, 5.0),
        ("부동소수점 오차 (0.1+0.2 와 0.3)", 0.1 + 0.2, 0.3),
        ("아주 작은 차이 (1e-12)", 1.0, 1.0 + 1e-12),
        ("A가 더 큼", 5.0, 3.0),
        ("B가 더 큼", 3.0, 5.0),
    ]
    print("EPSILON = {}  (abs(a-b) < EPSILON 이면 동점)".format(EPSILON))
    for title, score_a, score_b in pairs:
        print("\n- {}".format(title))
        print("  a={!r}, b={!r}".format(score_a, score_b))
        print(
            "  a == b 단순 비교: {} | abs(a-b) = {:.3e} | epsilon 판정: {}".format(
                score_a == score_b, abs(score_a - score_b), judge(score_a, score_b)
            )
        )


def analyze_mac_cases(data):
    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("'filters'와 'patterns'는 모두 객체여야 합니다.")

    filters_by_size = {}
    for size in (5, 13, 25):
        filter_key = "size_{}".format(size)
        try:
            if filter_key not in filters:
                raise ValueError("{}가 없습니다.".format(filter_key))
            filters_by_size[size] = normalize_filter_set(filters[filter_key], size)
        except (ValueError, TypeError) as error:
            print("필터 로드 실패: {} - {}".format(filter_key, error))

    total = 0
    passed = 0
    failures = []
    for case_id, case in patterns.items():
        total += 1
        print("\n--- {} ---".format(case_id))
        try:
            match = (
                SIZE_KEY_PATTERN.fullmatch(case_id)
                if isinstance(case_id, str)
                else None
            )
            if match is None:
                raise ValueError("케이스 키 형식이 size_{N}_{idx}와 다릅니다.")
            size = int(match.group(1))
            if size not in filters_by_size:
                raise ValueError("size_{} 필터를 사용할 수 없습니다.".format(size))
            if not isinstance(case, dict):
                raise ValueError("케이스 구조가 객체여야 합니다.")

            pattern = case.get("input")
            validate_matrix(pattern, size)
            expected = normalize_label(case.get("expected"))
            if expected is None:
                raise ValueError("expected 라벨을 Cross 또는 X로 변환할 수 없습니다.")

            cross_score = mac(pattern, filters_by_size[size]["Cross"])
            x_score = mac(pattern, filters_by_size[size]["X"])
            result = judge_labels(cross_score, x_score)
            status = "PASS" if result == expected else "FAIL"
            print("Cross 점수: {}".format(cross_score))
            print("X 점수: {}".format(x_score))
            print("판정: {} | expected: {} | {}".format(result, expected, status))
            if status == "PASS":
                passed += 1
            else:
                reason = (
                    "동점(UNDECIDED) 처리 규칙"
                    if result == "UNDECIDED"
                    else "판정 결과와 expected 불일치"
                )
                failures.append((case_id, reason))
        except (ValueError, TypeError, KeyError) as error:
            print("판정: FAIL")
            print("사유: 스키마/데이터 오류: {}".format(error))
            failures.append((case_id, "스키마/데이터 오류: {}".format(error)))

    print("\n결과 요약")
    print("총 테스트: {}개 / PASS: {}개 / FAIL: {}개".format(
        total, passed, total - passed
    ))
    if failures:
        print("실패 케이스:")
        for case_id, reason in failures:
            print("- {}: {}".format(case_id, reason))
    return {"total": total, "passed": passed, "failed": total - passed}


def print_standalone_code(functions, main_lines):
    print("\n" + "-" * 55)
    print("\n코딩 내용 (이 부분만 복사해도 단독 실행됩니다. data.json이 필요한 문제는 같은 폴더에 두세요):")
    print("import json")
    print("import math")
    print("import os")
    print("import re")
    print()
    print("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
    print("EPSILON = {}".format(EPSILON))
    print('SIZE_KEY_PATTERN = re.compile(r"^size_(\\d+)_(.+)$")')
    print()
    for function in functions:
        print(inspect.getsource(function).rstrip())
        print()
    for line in main_lines:
        print(line)
    print()
    print()
    print('if __name__ == "__main__":')
    print("    main()")
    print("-" * 55)


MAC_CORE_FUNCTIONS = (
    validate_matrix,
    _validate_matrix_position,
    set_matrix_value,
    get_matrix_value,
    mac,
)


def wait_for_menu(problem_name):
    try:
        input("\n[{}] 완료! Enter를 누르면 메뉴로 돌아갑니다.".format(problem_name))
    except EOFError:
        print("입력이 끝나 프로그램을 종료합니다.")


def run_mac_basic_mode():
    print("문제 내용:")
    print("입력 패턴과 필터를 위치별로 곱하고 모두 더하는 MAC 연산을 구현합니다.")
    print("NumPy 같은 외부 라이브러리 없이 반복문으로 직접 계산하고, 점수(float 가능)를 반환합니다.")
    print("\n[그림 보는 법]")
    print("패턴과 필터에서 같은 위치(행, 열)의 값끼리 곱해 '위치별 곱' 칸에 적습니다.")
    print("그 칸의 값을 모두 더한 것이 MAC 점수입니다.")
    demo_mac_examples()
    print_standalone_code(
        MAC_CORE_FUNCTIONS + (print_mac_grid, demo_mac_examples),
        ["def main():", "    demo_mac_examples()"],
    )
    return True


def run_epsilon_mode():
    print("문제 내용:")
    print("두 점수의 차이 abs(a-b)가 EPSILON(1e-9)보다 작으면 동점(UNDECIDED)으로 봅니다.")
    print("부동소수점 오차(예: 0.1+0.2 와 0.3) 때문에 같은 점수가 다르게 판정되는 일을 막습니다.")
    demo_epsilon_policy()
    print_standalone_code(
        (judge, demo_epsilon_policy),
        ["def main():", "    demo_epsilon_policy()"],
    )
    return True


def run_mac_judge_mode():
    print("문제 내용:")
    print("data.json의 각 패턴에 대해 Cross 필터 점수, X 필터 점수, 판정(Cross/X/UNDECIDED)을 출력합니다.")
    print("판정이 expected와 같으면 PASS, 다르면 FAIL입니다.")
    print("스키마/크기 오류가 있어도 중단하지 않고 해당 케이스만 FAIL 처리합니다.")
    try:
        data = load_json()
        analyze_mac_cases(data)
        print_standalone_code(
            MAC_CORE_FUNCTIONS
            + (
                normalize_label,
                normalize_filter_set,
                judge_labels,
                load_json,
                analyze_mac_cases,
            ),
            [
                "def main():",
                "    try:",
                "        data = load_json()",
                "        analyze_mac_cases(data)",
                "    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:",
                '        print("실행 오류: {}".format(error))',
            ],
        )
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def print_performance_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용 (이 부분만 복사해도 단독 실행됩니다):")
    print("import math")
    print("import time")
    print()
    print("REPEAT_COUNT = {}".format(REPEAT_COUNT))
    print("MAX_PATTERN_SIZE = {}".format(MAX_PATTERN_SIZE))
    print("SIZES = {}".format(SIZES))
    print()
    for function in (
        validate_matrix,
        _validate_matrix_position,
        set_matrix_value,
        get_matrix_value,
        mac,
        normalize_label,
        generate_pattern,
        make_benchmark_matrices,
        timed_average,
        print_performance_report,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("def main():")
    print("    print_performance_report()")
    print()
    print()
    print('if __name__ == "__main__":')
    print("    main()")
    print("-" * 55)


def run_performance_mode():
    print("문제 내용:")
    print("크기별(N×N) MAC 연산 시간을 ms 단위로 측정합니다.")
    print("측정 기준: 크기마다 MAC 연산을 {}회 반복 측정한 뒤 평균을 구합니다.".format(REPEAT_COUNT))
    print("입력/출력/파일 읽기 시간은 제외하고, mac() 함수 호출 구간만 측정합니다.")
    print("(패턴·필터 생성은 측정 전에 끝내며, 시간은 time.perf_counter_ns()로 잽니다.)")
    print("표에는 크기(N×N) / 평균 시간(ms) / 연산 횟수(N²)를 출력합니다.\n")
    print_performance_report()
    print_performance_code()
    return True


def collect_results(data):
    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("'filters'와 'patterns'는 모두 객체여야 합니다.")

    filters_by_size = {}
    filter_errors = {}
    for size in (5, 13, 25):
        filter_key = "size_{}".format(size)
        try:
            if filter_key not in filters:
                raise ValueError("{}가 없습니다.".format(filter_key))
            filters_by_size[size] = normalize_filter_set(filters[filter_key], size)
        except (ValueError, TypeError) as error:
            filter_errors[size] = str(error)

    total = 0
    passed = 0
    failures = []
    for case_id, case in patterns.items():
        total += 1
        try:
            match = (
                SIZE_KEY_PATTERN.fullmatch(case_id)
                if isinstance(case_id, str)
                else None
            )
            if match is None:
                raise ValueError("케이스 키 형식이 size_{N}_{idx}와 다릅니다.")
            size = int(match.group(1))
            if size not in filters_by_size:
                raise ValueError(
                    filter_errors.get(size, "size_{} 필터를 사용할 수 없습니다.".format(size))
                )
            if not isinstance(case, dict):
                raise ValueError("케이스 구조가 객체여야 합니다.")

            pattern = case.get("input")
            validate_matrix(pattern, size)
            expected = normalize_label(case.get("expected"))
            if expected is None:
                raise ValueError("expected 라벨을 Cross 또는 X로 변환할 수 없습니다.")

            cross_score = mac(pattern, filters_by_size[size]["Cross"])
            x_score = mac(pattern, filters_by_size[size]["X"])
            result = judge_labels(cross_score, x_score)
            if result == expected:
                passed += 1
            elif result == "UNDECIDED":
                failures.append((case_id, "동점(UNDECIDED) 처리 규칙"))
            else:
                failures.append(
                    (case_id, "판정 {} / expected {} 불일치".format(result, expected))
                )
        except (ValueError, TypeError, KeyError) as error:
            failures.append((str(case_id), "스키마/데이터 오류: {}".format(error)))

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "failures": failures,
    }


def print_result_report(report):
    print("\n[결과 리포트]")
    print("전체 테스트 수: {}개".format(report["total"]))
    print("통과 수: {}개".format(report["passed"]))
    print("실패 수: {}개".format(report["failed"]))
    if report["failures"]:
        print("실패 케이스 목록:")
        for case_id, reason in report["failures"]:
            print("- {}: {}".format(case_id, reason))
    else:
        print("실패 케이스: 없음")


def run_result_report_mode():
    print("문제 내용:")
    print("data.json 모드 수행 결과를 콘솔에 요약 리포트로 출력합니다.")
    print("전체 테스트 수 / 통과 수 / 실패 수를 출력합니다.")
    print("실패 케이스가 있으면 케이스 식별자와 실패 사유(요약)를 함께 출력합니다.")
    try:
        data = load_json()
        print_result_report(collect_results(data))
        print_standalone_code(
            MAC_CORE_FUNCTIONS
            + (
                normalize_label,
                normalize_filter_set,
                judge_labels,
                load_json,
                collect_results,
                print_result_report,
            ),
            [
                "def main():",
                "    try:",
                "        data = load_json()",
                "        print_result_report(collect_results(data))",
                "    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:",
                '        print("실행 오류: {}".format(error))',
            ],
        )
        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


def print_matrix_plain(matrix):
    for row in matrix:
        print(" ".join(str(value) for value in row))


def read_pattern_size():
    while True:
        try:
            text = input("패턴 크기 N 입력 (1~{}): ".format(MAX_PATTERN_SIZE)).strip()
        except EOFError:
            return None
        try:
            size = int(text)
            generate_pattern(size, "Cross")
            return size
        except ValueError as error:
            print("입력 오류: {}".format(error))


def run_pattern_demo(size):
    cross = generate_pattern(size, "Cross")
    x_pattern = generate_pattern(size, "X")
    print("\n[{}×{} Cross 패턴]".format(size, size))
    print_matrix_plain(cross)
    print("\n[{}×{} X 패턴]".format(size, size))
    print_matrix_plain(x_pattern)

    print("\n[생성 패턴 재활용: 모드 1과 같은 MAC 판정 (필터 A=Cross, 필터 B=X)]")
    for name, pattern in (("Cross 패턴", cross), ("X 패턴", x_pattern)):
        score_a = mac(pattern, cross)
        score_b = mac(pattern, x_pattern)
        print("{} 입력 -> A 점수 {} / B 점수 {} / 판정 {}".format(
            name, score_a, score_b, judge(score_a, score_b)
        ))

    flat_score = mac_flat(flatten_matrix(cross), flatten_matrix(x_pattern))
    print("\n[1차원 변환 검증]")
    print("2D mac = {} / 1D mac_flat = {} -> {}".format(
        mac(cross, x_pattern), flat_score,
        "일치" if abs(mac(cross, x_pattern) - flat_score) < EPSILON else "불일치"
    ))


def print_bonus_code():
    print("\n" + "-" * 55)
    print("\n코딩 내용 (이 부분만 복사해도 단독 실행됩니다):")
    print("import math")
    print("import time")
    print()
    print("EPSILON = {}".format(EPSILON))
    print("REPEAT_COUNT = {}".format(REPEAT_COUNT))
    print("MAX_PATTERN_SIZE = {}".format(MAX_PATTERN_SIZE))
    print("SIZES = {}".format(SIZES))
    print()
    for function in (
        validate_matrix,
        _validate_matrix_position,
        set_matrix_value,
        get_matrix_value,
        mac,
        normalize_label,
        generate_pattern,
        flatten_matrix,
        mac_flat,
        make_benchmark_matrices,
        timed_average,
        compare_memory_access,
        print_matrix_plain,
    ):
        print(inspect.getsource(function).rstrip())
        print()
    print("def main():")
    print('    size = int(input("패턴 크기 N 입력 (1~{}): ".format(MAX_PATTERN_SIZE)))')
    print('    print("[Cross 패턴]")')
    print('    print_matrix_plain(generate_pattern(size, "Cross"))')
    print('    print("[X 패턴]")')
    print('    print_matrix_plain(generate_pattern(size, "X"))')
    print("    sizes = SIZES if size in SIZES else SIZES + (size,)")
    print("    compare_memory_access(sizes)")
    print()
    print()
    print('if __name__ == "__main__":')
    print("    main()")
    print("-" * 55)


def run_bonus_mode():
    print("문제 내용:")
    print("1) 패턴 생성기: 크기 N을 입력하면 N×N Cross와 X 패턴을 자동 생성합니다.")
    print("   생성한 패턴은 MAC 판정(모드 1)과 성능 분석(7-4)에 그대로 재활용합니다.")
    print("2) 메모리 접근 최적화: 2차원 배열을 길이 N²의 1차원 배열로 펴서 계산합니다.")
    print("   동일 입력, 동일 반복 횟수({}회)로 최적화 전(2D)/후(1D) 시간을 비교합니다.".format(REPEAT_COUNT))
    print()
    size = read_pattern_size()
    if size is None:
        print("입력이 끝나 문제 9를 종료합니다.")
        return False
    try:
        run_pattern_demo(size)
        sizes = SIZES if size in SIZES else SIZES + (size,)
        print("\n[최적화 전/후 성능 비교 (2D vs 1D, 각 {}회 평균)]".format(REPEAT_COUNT))
        compare_memory_access(sizes)
        print("\n참고: 두 방식 모두 값 검사를 하므로, 배율에는 2D의 함수 호출/검사 시간 차이도 포함됩니다.")
        print_bonus_code()
        return True
    except (ValueError, TypeError) as error:
        print("실행 오류: {}".format(error))
        return False


PROGRAM_TITLE = "Mini NPU 시뮬레이터 - MAC 연산으로 배우는 패턴 판정"
CHECK_MARK = "✔"

MENU_ITEMS = (
    ("1", "사용자 입력 (3×3)"),
    ("2", "data.json 분석"),
    ("3", "data.json 데이터 구조"),
    ("4", "입력처리 (3×3)"),
    ("5", "JSON 로드 및 스키마 검증"),
    ("6", "라벨 정규화(표준화)"),
    ("7-1", "MAC 연산 구현 (반복문)"),
    ("7-2", "점수 비교 정책 (epsilon)"),
    ("7-3", "패턴별 Cross/X 점수 판정과 PASS/FAIL"),
    ("7-4", "성능 분석 (측정 기준 포함)"),
    ("8", "결과 리포트"),
    ("9", "보너스 과제 (패턴 생성기 + 메모리 접근 최적화)"),
)

KEY_POINTS = {
    "1": (
        "MAC은 같은 위치의 두 값을 곱하고, 그 곱을 모두 더하는 연산입니다.",
        "필터와 패턴이 닮을수록 곱해서 더한 점수가 커집니다.",
        "점수가 더 높은 필터를 패턴과 더 닮은 쪽으로 판정합니다.",
        "사용자는 숫자를 잘못 넣을 수 있으므로, 개수와 숫자 여부를 검사하고 틀리면 다시 입력하게 합니다.",
    ),
    "2": (
        "JSON은 필터와 패턴을 파일로 저장해 둔 데이터입니다.",
        "패턴마다 Cross 필터 점수와 X 필터 점수를 구해 더 높은 쪽으로 판정합니다.",
        "판정이 expected(정답)와 같으면 PASS, 다르면 FAIL입니다.",
    ),
    "3": (
        "프로그램을 만들기 전에 데이터가 어떤 모양인지부터 파악합니다.",
        "filters는 크기별(size_N) 필터 묶음, patterns는 size_N_번호 형식의 케이스 묶음입니다.",
        "키 이름에서 크기 N을 뽑아 알맞은 필터를 고르는 규칙의 바탕이 됩니다.",
    ),
    "4": (
        "사용자 입력은 언제든 틀릴 수 있다고 가정하고 만듭니다.",
        "행 수, 열마다 숫자 개수, 숫자가 아닌 값을 모두 검사해 오류를 한꺼번에 알려 줍니다.",
        "오류가 있어도 프로그램이 멈추지 않고 다시 입력하게 하는 것이 핵심입니다.",
    ),
    "5": (
        "스키마는 데이터가 지켜야 할 모양(형식)에 대한 약속입니다.",
        "케이스 하나가 잘못돼도 전체를 멈추지 않고, 그 케이스만 FAIL로 처리합니다.",
        "try/except로 오류를 잡아 나머지 케이스를 계속 검사합니다.",
    ),
    "6": (
        "데이터마다 '+', 'cross', 'x'처럼 표기가 다를 수 있습니다.",
        "프로그램 안에서는 표준 라벨 Cross, X 두 가지로 통일해서 사용합니다.",
        "표기를 먼저 통일해야 expected와 판정 결과를 정확히 비교할 수 있습니다.",
    ),
    "7-1": (
        "같은 위치끼리 곱한 값을 모두 더하는 일을 이중 반복문(행, 열)으로 직접 구현합니다.",
        "NumPy 없이 계산 과정을 직접 보면서 연산 원리를 이해하는 것이 목적입니다.",
        "N×N 행렬이면 곱셈과 덧셈이 N² 번 일어납니다.",
    ),
    "7-2": (
        "컴퓨터의 소수 계산에는 아주 작은 오차가 생깁니다. 예: 0.1+0.2 는 0.3과 정확히 같지 않습니다.",
        "그래서 두 점수를 == 로 비교하지 않고, 차이가 EPSILON(1e-9)보다 작으면 같은 점수로 봅니다.",
        "같은 점수라면 어느 쪽이라고 정할 수 없으므로 UNDECIDED(판정 불가)로 처리합니다.",
    ),
    "7-3": (
        "한 패턴을 Cross 필터와 X 필터에 각각 적용해 점수를 구합니다.",
        "점수가 더 높은 쪽이 판정 결과이고, 점수가 같으면 UNDECIDED입니다.",
        "판정이 expected와 같으면 PASS, 다르거나 데이터 오류가 있으면 FAIL이며, 오류가 있어도 다음 케이스를 계속합니다.",
    ),
    "7-4": (
        "시간은 한 번만 재면 들쭉날쭉하므로, 여러 번(10회) 반복해서 평균을 냅니다.",
        "입출력 시간은 빼고, MAC 연산 함수를 부르는 구간만 측정합니다.",
        "크기가 커지면 연산 횟수가 N²으로 늘어나므로 시간도 함께 늘어납니다.",
    ),
    "8": (
        "개별 결과를 모아 전체/통과/실패 수로 요약하면 상태를 한눈에 알 수 있습니다.",
        "실패한 케이스는 식별자와 사유를 함께 보여 주어야 어디를 고칠지 바로 알 수 있습니다.",
    ),
    "9": (
        "2차원 배열을 길이 N²의 1차원 배열로 펴면 값에 접근하는 방식이 단순해집니다.",
        "패턴 생성기는 크기 N만 알면 Cross/X 패턴을 자동으로 만들어 판정과 성능 분석에 다시 씁니다.",
        "최적화 전/후는 같은 입력, 같은 반복 횟수로 비교해야 공정합니다.",
        "시간 차이에는 접근 방식뿐 아니라 검사 시간의 차이도 섞여 있습니다.",
    ),
}


def begin_problem(key):
    title = dict(MENU_ITEMS)[key]
    clear_screen()
    print("\n[문제 {}] {}".format(key, title))
    print("\n[핵심 설명]")
    for line in KEY_POINTS[key]:
        print("- {}".format(line))
    print()


def print_banner(done_count, total_count):
    print("=" * 60)
    print("  {}".format(PROGRAM_TITLE))
    print("=" * 60)
    bar = "■" * done_count + "□" * (total_count - done_count)
    print("학습 진행: {}/{}  {}".format(done_count, total_count, bar))


def print_menu(completed):
    print("\n[문제 선택]  ({} = 완료한 문제)".format(CHECK_MARK))
    for key, title in MENU_ITEMS:
        mark = CHECK_MARK if key in completed else " "
        print(" [{}] {:<4} {}".format(mark, key + ".", title))
    print("     0.   종료")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runners = {
        "1": run_user_mode,
        "2": run_json_mode,
        "3": run_data_structure_mode,
        "4": run_input_processing_mode,
        "5": run_schema_validation_mode,
        "6": run_label_normalization_mode,
        "7-1": run_mac_basic_mode,
        "7-2": run_epsilon_mode,
        "7-3": run_mac_judge_mode,
        "7-4": run_performance_mode,
        "8": run_result_report_mode,
        "9": run_bonus_mode,
    }
    completed = set()
    notice = ""
    while True:
        clear_screen()
        print_banner(len(completed), len(MENU_ITEMS))
        print_menu(completed)
        if len(completed) == len(MENU_ITEMS):
            print("\n모든 문제를 마쳤습니다. 수고하셨습니다!")
        if notice:
            print("\n" + notice)
            notice = ""
        try:
            choice = input("\n선택: ").strip()
        except EOFError:
            print("입력이 끝나 프로그램을 종료합니다.")
            break
        if choice == "0":
            print("프로그램을 종료합니다.")
            break
        runner = runners.get(choice)
        if runner is None:
            notice = "잘못된 선택입니다. 메뉴에 있는 번호를 입력하세요."
            continue
        begin_problem(choice)
        if runner():
            completed.add(choice)
        wait_for_menu("문제 {}".format(choice))


if __name__ == "__main__":
    main()