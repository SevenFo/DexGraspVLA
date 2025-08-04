import cProfile
import pstats
import io
from functools import wraps

def profile_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        breakpoint()
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # 输出到字符串而不是stdout
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print(f"\n=== Profile results for {func.__name__} ===")
        print(s.getvalue())
        
        return result
    return wrapper

# 修复后的类级别的装饰器
def profile_class(include_private=False, exclude=None):
    """
    装饰整个类的方法
    :param include_private: 是否包含私有方法（以_开头）
    :param exclude: 要排除的方法列表
    """
    def decorator(cls):
        exclude_list = exclude or []
        # 添加一些不能装饰的特殊属性
        special_attrs = {'__class__', '__dict__', '__weakref__', '__doc__', '__module__', '__annotations__'}
        
        for attr_name in dir(cls):
            # 跳过特殊属性和排除的方法
            if attr_name in special_attrs or attr_name in exclude_list:
                continue
            
            # 根据设置决定是否包含私有方法
            if not include_private and attr_name.startswith('_'):
                continue
                
            attr = getattr(cls, attr_name)
            # 只装饰可调用的属性（方法），并且确保不是内置方法
            if callable(attr) and hasattr(attr, '__func__') or (callable(attr) and not attr_name.startswith('__')):
                # 额外检查，确保是实例方法而不是特殊属性
                if not (attr_name.startswith('__') and attr_name.endswith('__')):
                    setattr(cls, attr_name, profile_func(attr))
                elif attr_name in ['__init__', '__str__', '__repr__']:  # 只装饰一些安全的特殊方法
                    setattr(cls, attr_name, profile_func(attr))
        return cls
    return decorator

def profile_methods(*method_names):
    """只装饰指定的方法"""
    def decorator(cls):
        for method_name in method_names:
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                if callable(method):
                    setattr(cls, method_name, profile_func(method))
        return cls
    return decorator


if __name__ == '__main__':
        
    # 使用示例
    @profile_func
    def your_slow_function():
        # 你的代码
        for i in range(1000000):
            range(2)
            another_function()
            pass

    # @profile_func
    def another_function(data = None):
        if not data:
            return range(2)
        # 你的代码
        result = []
        for item in data:
            result.append(item * 2)
        return result
    
    # 测试用例1: 基本函数profile
    print("=== 测试用例1: 基本函数profile ===")
    # your_slow_function()  # 暂时注释掉避免干扰其他测试
    # another_function(range(1000))

    # 测试用例2: 类级别的装饰器测试
    print("\n=== 测试用例2: profile_class 装饰器 ===")
    
    @profile_class(exclude=['helper_method'])
    class TestClass1:
        def __init__(self):
            self.data = list(range(1000))
        
        def slow_method(self):
            result = []
            for i in range(10000):
                result.extend(self.helper_method())
            return result
        
        def helper_method(self):
            return [x * 2 for x in range(3)]
        
        def fast_method(self):
            return "quick"
    
    # 运行测试
    obj1 = TestClass1()
    obj1.slow_method()
    obj1.fast_method()

    # 测试用例3: 指定方法装饰器测试
    print("\n=== 测试用例3: profile_methods 装饰器 ===")
    
    @profile_methods('expensive_method', 'slow_init','__init__')
    class TestClass2:
        def __init__(self):
            self.data = []
        
        def slow_init(self):
            for i in range(10000):
                self.data.append(i * 2)
        
        def expensive_method(self):
            result = []
            for i in range(50000):
                result.append(i ** 2)
            return result
        
        def fast_method(self):
            return sum(self.data)
    
    # 运行测试
    obj2 = TestClass2()
    # obj2.slow_init()      # 这个不会被profile
    obj2.expensive_method()  # 这个会被profile
    # obj2.fast_method()     # 这个不会被profile

    # 测试用例4: 包含私有方法的类装饰器
    print("\n=== 测试用例4: 包含私有方法的profile ===")
    
    @profile_class(include_private=True, exclude=['__init__'])
    class TestClass3:
        def __init__(self):
            self._private_data = list(range(100))
        
        def public_method(self):
            return self._private_helper()
        
        def _private_helper(self):
            return [x * 3 for x in self._private_data]
    
    # 运行测试
    obj3 = TestClass3()
    result = obj3.public_method()  # 会同时profile _private_helper
    print(f"Public method result: {result[:5]}...")  # 显示部分结果
