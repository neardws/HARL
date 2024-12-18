import sys
sys.path.append(r"/root/HARL/")
from objects.mobility_object import mobility
from utilities.time_calculation import transform_str_data_time_into_timestamp_ms, transform_str_data_time_into_timestamp
from typing import List, Optional, Tuple
import pandas as pd
import random


class TrajectoriesProcessing(object):
    
    def __init__(
        self,
        file_name_key : Optional[str] = "",
        vehicle_number : Optional[int] = 10,  # 定义车辆数目
        start_time : Optional[str] = "2005-04-13 16:00:00",  # 定义开始时间
        slot_length : Optional[int] = 300,
        slection_way : Optional[str] = "random",  # 定义车辆选择方式, 'random', 'max_duration', 'first'
        filling_way : Optional[str] = 'linear',  # 定义缺失值填充方式
        chunk_size: Optional[int] = 100000,   # 定义每块的大小（根据实际情况进行调整）
        is_ms: Optional[bool] = True,
    ) -> None:
        self._file_names : dict = {}
        # I-80-Emeryville-CA-1650feet
        self._file_names['I-80-Emeryville-CA-1650feet-0400pm-0415pm'] = 'trajectories_files/I-80-Emeryville-CA-1650feet/i-80-vehicle-trajectory-data/vehicle-trajectory-data/0400pm-0415pm/trajectories-0400-0415.csv'
        self._file_names['I-80-Emeryville-CA-1650feet-0500pm-0515pm'] = 'trajectories_files/I-80-Emeryville-CA-1650feet/i-80-vehicle-trajectory-data/vehicle-trajectory-data/0500pm-0515pm/trajectories-0500-0515.csv'
        self._file_names['I-80-Emeryville-CA-1650feet-0515pm-0530pm'] = 'trajectories_files/I-80-Emeryville-CA-1650feet/i-80-vehicle-trajectory-data/vehicle-trajectory-data/0515pm-0530pm/trajectories-0515-0530.csv'
        # Lankershim-Boulevard-LosAngeles-CA-1600feet
        self._file_names['Lankershim-Boulevard-LosAngeles-CA-1600feet'] = 'trajectories_files/Lankershim-Boulevard-LosAngeles-CA-1600feet/NGSIM__Lankershim_Vehicle_Trajectories.csv'
        # Peachtree-Street-Atlanta-GA-2100feet
        self._file_names['Peachtree-Street-Atlanta-GA-2100feet'] = 'trajectories_files/Peachtree-Street-Atlanta-GA-2100feet/NGSIM_Peachtree_Vehicle_Trajectories.csv'
        # US-101-SanDiego-CA-1500feet
        self._file_names['US-101-SanDiego-CA-1500feet-0750am-0805am'] = 'trajectories_files/US-101-LosAngeles-CA-2100feet/us-101-vehicle-trajectory-data/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.csv'
        self._file_names['US-101-SanDiego-CA-1500feet-0805am-0820am'] = 'trajectories_files/US-101-LosAngeles-CA-2100feet/us-101-vehicle-trajectory-data/vehicle-trajectory-data/0805am-0820am/trajectories-0805am-0820am.csv'
        self._file_names['US-101-SanDiego-CA-1500feet-0820am-0835am'] = 'trajectories_files/US-101-LosAngeles-CA-2100feet/us-101-vehicle-trajectory-data/vehicle-trajectory-data/0820am-0835am/trajectories-0820am-0835am.csv'
        
        self._start_times : dict = {}
        self._end_times : dict = {}
        self._start_times['I-80-Emeryville-CA-1650feet-0400pm-0415pm'] = '2005-04-13 16:00:00'
        self._end_times['I-80-Emeryville-CA-1650feet-0400pm-0415pm'] = '2005-04-13 16:15:00'
        
        self._start_times['I-80-Emeryville-CA-1650feet-0500pm-0515pm'] = '2005-04-13 17:00:00'
        self._end_times['I-80-Emeryville-CA-1650feet-0500pm-0515pm'] = '2005-04-13 17:15:00'
        
        self._start_times['I-80-Emeryville-CA-1650feet-0515pm-0530pm'] = '2005-04-13 17:15:00'
        self._end_times['I-80-Emeryville-CA-1650feet-0515pm-0530pm'] = '2005-04-13 17:30:00'
        
        self._start_times['Lankershim-Boulevard-LosAngeles-CA-1600feet'] = '2005-06-16 08:30:00'
        self._end_times['Lankershim-Boulevard-LosAngeles-CA-1600feet'] = '2005-06-16 09:00:00'
        
        self._start_times['Peachtree-Street-Atlanta-GA-2100feet'] = '2006-11-08 16:00:00'
        
        self._start_times['US-101-SanDiego-CA-1500feet-0750am-0805am'] = '2005-07-19 07:50:00'
        self._end_times['US-101-SanDiego-CA-1500feet-0750am-0805am'] = '2005-07-19 08:05:00'
        
        self._start_times['US-101-SanDiego-CA-1500feet-0805am-0820am'] = '2005-07-19 08:05:00'
        self._end_times['US-101-SanDiego-CA-1500feet-0805am-0820am'] = '2005-07-19 08:20:00'
        
        self._start_times['US-101-SanDiego-CA-1500feet-0820am-0835am'] = '2005-07-19 08:20:00'
        self._end_times['US-101-SanDiego-CA-1500feet-0820am-0835am'] = '2005-07-19 08:35:00'
        
        self._file_name_key : Optional[str] = file_name_key
        self._is_ms : bool = is_ms
        
        if self._file_name_key not in self._file_names:
            raise ValueError('file_name_key not found')
        else:
            if self._file_name_key == 'Peachtree-Street-Atlanta-GA-2100feet':
                self._is_ms = False
        
        self._chunk_size : Optional[int] = chunk_size
        
        self._vehicle_number : Optional[int] = vehicle_number
        self._slection_way : Optional[str] = slection_way
        self._filling_way : Optional[str] = filling_way
        
        self._vehicle_IDs : List[int] = []
        self._selected_vehicle_IDs : List[int] = []
        
        self._selected_data : pd.DataFrame = pd.DataFrame()
        self._transfered_data : pd.DataFrame = pd.DataFrame()        
        self._vehicle_mobilities : List[List[mobility]] = []
        
        # 初始化一个空DataFrame用于存储所有车辆的轨迹数据
        self._all_data = pd.DataFrame()
        self._min_global_time = 0
        
        self._start_time : Optional[str] = start_time
        self._slot_time_length : Optional[int] = slot_length
        
        self._start_time_timestamp_ms : int = 0
        self._end_time_timestamp_ms : int = 0
        
    def get_max_and_min_global_time(self) -> Tuple[int, int]:
        return self._all_data['Global_Time'].max(), self._all_data['Global_Time'].min()
    
    def get_file_name_key(self) -> str:
        return self._file_name_key
    
    def get_file_name(self) -> str:
        if self._file_name_key not in self._file_names:
            raise ValueError('file_name not found')
        return self._file_names[self._file_name_key]
    
    def get_vehicle_number(self) -> int:
        return self._vehicle_number
    
    def get_slection_way(self) -> str:
        return self._slection_way
    
    def get_filling_way(self) -> str:
        return self._filling_way
    
    def get_vehicle_IDs(self) -> List[int]:
        return self._vehicle_IDs
    
    def get_selected_vehicle_IDs(self) -> List[int]:
        return self._selected_vehicle_IDs
    
    def get_selected_data(self) -> pd.DataFrame:
        return self._selected_data
    
    def get_vehicle_mobilities(self) -> List[List[mobility]]:
        return self._vehicle_mobilities
    
    def set_file_name_key(self, file_name_key: str) -> None:
        self._file_name_key = file_name_key
        return None
    
    def set_vehicle_number(self, vehicle_number: int) -> None:
        self._vehicle_number = vehicle_number
        return None
    
    def set_slection_way(self, slection_way: str) -> None:
        self._slection_way = slection_way
        return None
    
    def set_filling_way(self, filling_way: str) -> None:
        self._filling_way = filling_way
        return None
    
    def set_start_time(self, start_time: str) -> None:
        self._start_time = start_time
        return None
    
    def obtain_start_end_time_stamp_ms(self) -> None:
        if self._is_ms:
            self._start_time_timestamp_ms = transform_str_data_time_into_timestamp_ms(self._start_time)
            self._end_time_timestamp_ms = self._start_time_timestamp_ms + self._slot_time_length * 1000
        else:
            self._start_time_timestamp_ms = transform_str_data_time_into_timestamp(self._start_time)
            self._end_time_timestamp_ms = self._start_time_timestamp_ms + self._slot_time_length
    
    """
    列名	         描述
    Vehicle_Id	    车辆识别号（根据进入该区域的时间升序），重复利用
    Frame_Id	    该条数据在某一时刻的帧（按开始时间升序），同一Vehicle_ID的帧号不会重复
    Total_Frame	    该车出现在此数据集的总帧数
    Global_Time	    时间戳（ms）
    Local_X	        车辆前部中心的横向（X）坐标，以英尺为单位，相对于截面在行驶方向上的最左侧边缘。
    Local_Y	        车辆前部中心的纵向（Y）坐标，以英尺为单位，相对于截面在行驶方向上的进入边缘。
                    以上两个	采集区域内的坐标，采集区域不同，坐标系不同，会有不同的零点
    Global_X,Y	    全局坐标，只有一个零点，可用作数据筛选
    v_length	    车辆长度（以英尺为单位）
    v_Width	        车辆长度（以英尺为单位）
    v_Class	        车辆类型：1-摩托车，2-汽车，3-卡车
    v_Vel	        车辆瞬时速度，以英尺/秒为单位
    v_Acc	        车辆的瞬时加速度，以英尺/秒为单位
    Lane_ID	        车辆的当前车道位置。 第1车道是最左边的车道； 第5车道是最右边的车道。
    O_Zone	        车辆的起点区域，即车辆进入跟踪系统的位置。 研究区域有11个起源，编号从101到111。有关更多详细信息，请参阅数据分析报告。
    D_Zone	        车辆的目的地区域，即车辆离开跟踪系统的地方。 研究区域中有10个目的地，从201到211编号。起点102是单向出口； 因此，没有关联的目标号码202。请参阅数据分析报告以获取更多详细信息。
    Int_ID	        车辆行驶的路口。 交叉点的编号为1到4，交叉点1位于最南端，交叉点4位于研究区域的最北端。 值为“ 0”表示该车辆不在交叉路口的附近，而是该车辆标识为Lankershim Boulevard的一段（下面的Section_ID）。 请参阅数据分析报告以获取更多详细信息。
    Section_ID	    车辆行驶的路段。 Lankershim Blvd分为五个部分（路口1的南部；路口1和2、2和3、3和4之间；路口4的北部）。 值为“ 0”表示该车辆未识别出Lankershim Boulevard的一段，并且该车辆紧邻交叉路口（上述Int_ID）。 请参阅数据分析报告以获取更多详细信息
    Direction	    车辆的行驶方向。 1-东行（EB），2-北行（NB），3-西行（WB），4-南行（SB）
    Movement	    车辆的运动。 1-通过（THE），2-左转（LEFT），3-右转（RT）。
    Preceding	    同道前车的车辆编号。数值为“0”表示没有前面的车辆-发生在研究段的末尾和出匝道
    Following	    在同一车道上跟随本车辆的车辆的车辆ID。 值“ 0”表示没有跟随的车辆-在研究部分的开头和匝道发生，
    Space_Headway	间距提供了车辆的前中心到前一辆车辆的前中心之间的距离。（英尺）
    Time_Headway	时间进度（以秒为单位）提供了从车辆的前中心（以车辆的速度）行进到前一辆车辆的前中心的时间。
    Location	    街道名称或高速公路名称
    """
    def read_csv(
        self, 
    ) -> None:
        try: 
            self.obtain_start_end_time_stamp_ms()
            # 使用迭代器逐块读取CSV文件
            reader = pd.read_csv(self.get_file_name(), chunksize=self._chunk_size)

            # 遍历每块数据
            for chunk in reader:
                # 得到指定开始结束时间段内的数据
                mask = (chunk['Global_Time'] >= self._start_time_timestamp_ms) & \
                    (chunk['Global_Time'] <= self._end_time_timestamp_ms)
                self._all_data = pd.concat([self._all_data, chunk[mask]])

            self._min_global_time = self._all_data['Global_Time'].min()
        except:
            raise ValueError('file_name not found')
    
    def read_all_csv(
        self,
    ) -> None:
        try: 
            # 使用迭代器逐块读取CSV文件
            reader = pd.read_csv(self.get_file_name(), chunksize=self._chunk_size)

            # 遍历每块数据
            for chunk in reader:
                self._all_data = pd.concat([self._all_data, chunk])

            self._min_global_time = self._all_data['Global_Time'].min()
        except:
            raise ValueError('file_name not found')
    
    def get_all_data(self):
        return self._all_data
    
    def select_vehicles(
        self,
    ) -> None:
        for vehicle_id, group in self._all_data.groupby('Vehicle_ID'):
            self._vehicle_IDs.append(vehicle_id)
        # 选择车辆
        if self._slection_way == 'random':
            # 随机选择车辆
            self._selected_vehicle_IDs = random.choices(self._vehicle_IDs, k=self._vehicle_number)
        elif self._slection_way == 'max_duration':
            # 选择轨迹最长的N辆车
            groups = self._all_data.groupby('Vehicle_ID')
            for group in groups:
                self._selected_vehicle_IDs.append(group[0])
            self._selected_vehicle_IDs.sort(key=lambda x: groups.get_group(x)['Global_Time'].count(), reverse=True)
            self._selected_vehicle_IDs = self._selected_vehicle_IDs[:self._vehicle_number]
        elif self._slection_way == 'first':
            # 选择前N辆车
            self._selected_vehicle_IDs = self._vehicle_IDs[:self._vehicle_number]
        else:
            raise ValueError('slection_way not found')
        self._selected_vehicle_IDs.sort()
        self._selected_data = self._all_data[self._all_data['Vehicle_ID'].isin(self._selected_vehicle_IDs)]
        return None
    
    def transfer_into_local_coordinates(
        self,
    ) -> Tuple[float, float, float, float]:
        groups = self._selected_data.groupby('Vehicle_ID')
        self._transfered_data = pd.DataFrame()
        for group in groups:
            # 对时间进行排序
            sorted_group = group[1].sort_values('Global_Time')
            # 将时间转化成相对毫秒
            sorted_group['Global_Time'] = sorted_group['Global_Time'] - self._min_global_time
            # 将坐标转换为公制单位
            sorted_group['Local_X'] = sorted_group['Local_X'] * 0.3048
            sorted_group['Local_Y'] = sorted_group['Local_Y'] * 0.3048
            sorted_group['v_Vel'] = sorted_group['v_Vel'] * 0.3048
            sorted_group['v_Acc'] = sorted_group['v_Acc'] * 0.3048
            self._transfered_data = pd.concat([self._transfered_data, sorted_group])
        min_map_x = self._transfered_data['Local_X'].min()
        min_map_y = self._transfered_data['Local_Y'].min()
        max_map_x = self._transfered_data['Local_X'].max()
        max_map_y = self._transfered_data['Local_Y'].max()
        return min_map_x, max_map_x, min_map_y, max_map_y
        
    def fill_missing_values(
        self,
    ) -> None:
        groups = self._transfered_data.groupby('Vehicle_ID')
        for group in groups:
            vehicle_mobility = []
            global_times = group[1]['Global_Time'].values
            # print("global_time", global_times)
            direction = 0
            try: 
                direction = group[1]['Direction'].values[0]
            except KeyError:
                #  1-东行（EB），2-北行（NB），3-西行（WB），4-南行（SB）
                local_x_difference = group[1]['Local_X'].values[len(global_times) - 1] - group[1]['Local_X'].values[0]
                local_y_difference = group[1]['Local_Y'].values[len(global_times) - 1] - group[1]['Local_Y'].values[0]
                if local_x_difference > 0 and local_x_difference > local_y_difference:
                    direction = 1
                elif local_y_difference > 0 and local_y_difference > local_x_difference:
                    direction = 2
                elif local_x_difference < 0 and - local_x_difference > local_y_difference:
                    direction = 3
                elif local_y_difference < 0 and - local_y_difference > local_x_difference:
                    direction = 4
                else:
                    raise ValueError('direction not found')
            for i in range(len(global_times)):
                global_time = global_times[i]
                local_x = group[1]['Local_X'].values[i]
                local_y = group[1]['Local_Y'].values[i]
                v_vel = group[1]['v_Vel'].values[i]
                v_acc = group[1]['v_Acc'].values[i]
                if self._is_ms and global_time % 1000 == 0:
                    global_time = int(global_time / 1000)
                    vehicle_mobility.append(
                        mobility(
                            x=local_x,
                            y=local_y,
                            speed=v_vel,
                            acceleration=v_acc,
                            direction=direction,
                            time=global_time,
                        )
                    )
                elif not self._is_ms:
                    vehicle_mobility.append(
                        mobility(
                            x=local_x,
                            y=local_y,
                            speed=v_vel,
                            acceleration=v_acc,
                            direction=direction,
                            time=global_time,
                        )
                    )
            self._vehicle_mobilities.append(vehicle_mobility)
            
        if self._filling_way == 'linear':
            for vehicle_index in range(len(self._vehicle_mobilities)):
                mobility_index = 0
                while mobility_index <= self._slot_time_length - 1:
                    # print("\nmobility_index", mobility_index)
                    # print("\nlen(self._vehicle_mobilities[vehicle_index]", len(self._vehicle_mobilities[vehicle_index]))
                    if mobility_index == 0 and self._vehicle_mobilities[vehicle_index][mobility_index].get_time() != 0:
                        # print("\nCase 1")
                        time_difference = self._vehicle_mobilities[vehicle_index][mobility_index].get_time()
                        local_x_difference = (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_x() - 
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_x()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_time()) * \
                            time_difference
                        local_y_difference = (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_y() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_y()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_time()) * \
                            time_difference
                        speed_difference = (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_speed() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_speed()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_time()) * \
                            time_difference
                        acceleration_difference = (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_acceleration() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() -
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_time()) * \
                            time_difference
                        direction = self._vehicle_mobilities[vehicle_index][mobility_index].get_direction()
                        for i in range(time_difference):
                            local_x = self._vehicle_mobilities[vehicle_index][mobility_index].get_x() + \
                                local_x_difference / time_difference * (i + 1)
                            local_y = self._vehicle_mobilities[vehicle_index][mobility_index].get_y() + \
                                local_y_difference / time_difference * (i + 1)
                            speed = self._vehicle_mobilities[vehicle_index][mobility_index].get_speed() + \
                                speed_difference / time_difference * (i + 1)
                            acceleration = self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration() + \
                                acceleration_difference / time_difference * (i + 1)
                            self._vehicle_mobilities[vehicle_index].insert(i, \
                                mobility(
                                    x=local_x,
                                    y=local_y,
                                    speed=speed,
                                    acceleration=acceleration,
                                    direction=direction,
                                    time=i,
                                )
                            )
                        mobility_index = mobility_index + time_difference + 1
                    elif mobility_index != len(self._vehicle_mobilities[vehicle_index]) - 1 and \
                        self._vehicle_mobilities[vehicle_index][mobility_index].get_time() \
                        != self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() - 1:
                        # print("\nCase 2")                        
                        time_difference = self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - 1
                        local_x_difference = self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_x() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_x()
                        local_y_difference = self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_y() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_y()
                        speed_difference = self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_speed() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_speed()
                        acceleration_difference = self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_acceleration() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration()
                        direction = self._vehicle_mobilities[vehicle_index][mobility_index].get_direction()
                        for i in range(time_difference):
                            # 
                            local_x = self._vehicle_mobilities[vehicle_index][mobility_index].get_x() + \
                                local_x_difference / time_difference * (i + 1)
                            local_y = self._vehicle_mobilities[vehicle_index][mobility_index].get_y() + \
                                local_y_difference / time_difference * (i + 1)
                            speed = self._vehicle_mobilities[vehicle_index][mobility_index].get_speed() + \
                                speed_difference / time_difference * (i + 1)
                            acceleration = self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration() + \
                                acceleration_difference / time_difference * (i + 1)
                            self._vehicle_mobilities[vehicle_index].insert(mobility_index + 1 + i, \
                                mobility(
                                    x=local_x,
                                    y=local_y,
                                    speed=speed,
                                    acceleration=acceleration,
                                    direction=direction,
                                    time=self._vehicle_mobilities[vehicle_index][mobility_index].get_time() + i + 1,
                                )
                            )
                        mobility_index = mobility_index + time_difference + 1
                    elif mobility_index == len(self._vehicle_mobilities[vehicle_index]) - 1 and \
                        self._vehicle_mobilities[vehicle_index][mobility_index].get_time() != self._slot_time_length - 1:
                        # print("\nCase 3")
                        time_difference = self._slot_time_length - self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - 1
                        local_x_difference = (self._vehicle_mobilities[vehicle_index][mobility_index].get_x() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_x()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_time()) * \
                            time_difference
                        local_y_difference = (self._vehicle_mobilities[vehicle_index][mobility_index].get_y() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_y()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_time()) * \
                            time_difference
                        speed_difference = (self._vehicle_mobilities[vehicle_index][mobility_index].get_speed() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_speed()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_time()) * \
                            time_difference
                        acceleration_difference = (self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_acceleration()) / \
                            (self._vehicle_mobilities[vehicle_index][mobility_index].get_time() - \
                            self._vehicle_mobilities[vehicle_index][mobility_index - 1].get_time()) * \
                            time_difference
                        direction = self._vehicle_mobilities[vehicle_index][mobility_index].get_direction()
                        for i in range(time_difference):
                            local_x = self._vehicle_mobilities[vehicle_index][mobility_index].get_x() + \
                                local_x_difference / time_difference * (i + 1)
                            local_y = self._vehicle_mobilities[vehicle_index][mobility_index].get_y() + \
                                local_y_difference / time_difference * (i + 1)
                            speed = self._vehicle_mobilities[vehicle_index][mobility_index].get_speed() + \
                                speed_difference / time_difference * (i + 1)
                            acceleration = self._vehicle_mobilities[vehicle_index][mobility_index].get_acceleration() + \
                                acceleration_difference / time_difference * (i + 1)
                            self._vehicle_mobilities[vehicle_index].insert(mobility_index + 1 + i, \
                                mobility(
                                    x=local_x,
                                    y=local_y,
                                    speed=speed,
                                    acceleration=acceleration,
                                    direction=direction,
                                    time=self._vehicle_mobilities[vehicle_index][mobility_index].get_time() + i + 1,
                                )
                            )
                        mobility_index = mobility_index + time_difference + 1
                    elif self._vehicle_mobilities[vehicle_index][mobility_index].get_time() != self._slot_time_length - 1 and \
                        self._vehicle_mobilities[vehicle_index][mobility_index].get_time() == self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time() - 1:
                        mobility_index = mobility_index + 1
                    elif self._vehicle_mobilities[vehicle_index][mobility_index].get_time() == self._slot_time_length - 1:
                        mobility_index = mobility_index + 1
                    else:
                        print("\nCase 4")
                        print("mobility_index", mobility_index)
                        print("len(self._vehicle_mobilities[vehicle_index]) - 1", len(self._vehicle_mobilities[vehicle_index]) - 1)
                        print("self._vehicle_mobilities[vehicle_index][mobility_index].get_time()", self._vehicle_mobilities[vehicle_index][mobility_index].get_time())
                        print("self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time()", self._vehicle_mobilities[vehicle_index][mobility_index + 1].get_time())
                        print("*" * 100)
        else:
            raise ValueError('filling_way not found')
        
        return None
    
    def processing(
        self
    ) -> Tuple[float, float, float, float]:
        self.read_csv()
        # print("read_csv finished")
        self.select_vehicles()
        # print("select_vehicles finished")
        min_map_x, max_map_x, min_map_y, max_map_y = self.transfer_into_local_coordinates()
        # print("transfer_into_local_coordinates finished")
        self.fill_missing_values()
        # print("fill_missing_values finished")
        # self.print_analysis(self._transfered_data)
        return min_map_x, max_map_x, min_map_y, max_map_y

    def print_analysis(self, data: pd.DataFrame) -> None:
        # 计算统计信息
        stats = data.groupby('Vehicle_ID').agg({
            'Global_X': ['min', 'max'],
            'Global_Y': ['min', 'max'],
            'Global_Time': 'count',
            'v_Vel': ['mean', 'max'],
            'v_Acc': 'mean'
        })

        # 格式化表格输出
        stats.columns = ['Min_X', 'Max_X', 'Min_Y', 'Max_Y', 'Track_Length', 'Avg_Speed', 'Max_Speed', 'Avg_Acc']
        stats.columns.name = 'Statistics'
        stats.index.name = 'Vehicle_ID'

        # 计算额外统计信息
        num_vehicles = data['Vehicle_ID'].nunique()
        overall_avg_speed = data['v_Vel'].mean()
        overall_max_speed = data['v_Vel'].max()
        overall_avg_acc = data['v_Acc'].mean()

        # 分析车辆比较多的时间段
        # data['Timestamp'] = pd.to_datetime(data['Global_Time'], unit='ms')
        # data['Hour'] = data['Timestamp'].dt.hour
        # vehicles_per_hour = data.groupby('Hour')['Vehicle_ID'].nunique()
        # peak_hour = vehicles_per_hour.idxmax()
        # peak_hour_vehicle_count = vehicles_per_hour.max()

        # # 找到最多车辆的1小时时间段内的5分钟细分
        # peak_hour_data = data[data['Hour'] == peak_hour]
        # peak_hour_data['5min'] = peak_hour_data['Timestamp'].dt.floor('5T')
        # vehicles_per_5min = peak_hour_data.groupby('5min')['Vehicle_ID'].nunique()
        # peak_5min = vehicles_per_5min.idxmax()
        # peak_5min_vehicle_count = vehicles_per_5min.max()

        # 打印统计信息
        print("Number of different Vehicle IDs:", num_vehicles)
        print("Overall average speed (feet/second):", overall_avg_speed)
        print("Overall maximum speed (feet/second):", overall_max_speed)
        print("Overall average acceleration (feet/second^2):", overall_avg_acc)
        # print("Peak hour (0-23):", peak_hour)
        # print("Number of vehicles during peak hour:", peak_hour_vehicle_count)
        # print("Peak 5-minute period within peak hour:", peak_5min)
        # print("Number of vehicles during peak 5-minute period:", peak_5min_vehicle_count)

        # 显示表格
        print(stats)

        # # 遍历所有车辆的轨迹数据并绘制轨迹
        # for vehicle_id, group in data.groupby('Vehicle_ID'):
        #     plt.plot(group['Local_X'], group['Local_Y'], label=f'Car {vehicle_id}')

        # # 添加图例和标签
        # plt.legend()
        # plt.xlabel('Local_X')
        # plt.ylabel('Local_Y')

        # # 显示图形
        # plt.show()
        
        return None
    
    
if __name__ == "__main__":
    trajectoriesProcessing = TrajectoriesProcessing(
        file_name_key='I-80-Emeryville-CA-1650feet-0400pm-0415pm',
        chunk_size=100000,
        start_time='2005-04-13 16:00:00',
    )
    trajectoriesProcessing.read_all_csv()
    trajectoriesProcessing.print_analysis(trajectoriesProcessing.get_all_data())
