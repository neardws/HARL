class mobility(object):
    '''the mobility of a vehicle is defined by its position, speed and direction'''
    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        acceleration: float,
        direction: float,  #  1-东行（EB），2-北行（NB），3-西行（WB），4-南行（SB）
        time : float,
    ) -> None:
        self._x : float = x
        self._y : float = y
        self._speed : float = speed
        self._acceleration : float = acceleration
        self._direction : float = direction
        self._time = time
        
    def get_x(self) -> float:
        return self._x
    
    def get_y(self) -> float:
        return self._y
    
    def get_speed(self) -> float:
        return self._speed
    
    def get_acceleration(self) -> float:
        return self._acceleration
    
    def get_direction(self) -> float:
        return self._direction
    
    def get_time(self) -> float:
        return self._time
    
    def __str__(self) -> str:
        return "x: " + str(self._x) + " y: " + str(self._y) + " speed: " + str(self._speed) + " acceleration: " + str(self._acceleration) + " direction: " + str(self._direction) + " time: " + str(self._time)